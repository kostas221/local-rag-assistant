"""Dynamic INT8 (fbgemm) στον cross-encoder: αξίζει, και τι κοστίζει στο gate;

ΓΙΑΤΙ ΞΑΝΑ INT8: το ONNX INT8 είχε μετρηθεί και ΑΠΟΡΡΙΦΘΕΙ (1.0x, μηδέν κέρδος).
Αυτό είναι ΑΛΛΟ μονοπάτι — quantize_dynamic του PyTorch με fbgemm kernels αντί
για onnxruntime — και σε πρόχειρη μέτρηση του forward pass έδωσε 1.36x. Ένα
απορριφθέν πείραμα δεν απορρίπτει μια διαφορετική υλοποίηση, αλλά ούτε την
εγκρίνει: γι' αυτό μετριέται εδώ πλήρως.

ΤΙ ΚΡΙΝΕΤΑΙ (με αυτή τη σειρά προτεραιότητας):
  1. TOP-1 FLIPS σε πραγματικά ζεύγη. Το page expansion τρέφεται από την
     ΚΑΤΑΤΑΞΗ· ένα flip = άλλη σελίδα στο context = άλλη απάντηση.
  2. ΠΕΡΙΘΩΡΙΟ ΤΟΥ GATE. Το MIN_RERANK_SCORE=-2.0 βαθμονομήθηκε πάνω σε ωμά
     logits: χειρότερο in-corpus -1.797, καλύτερο out-of-corpus -2.685. Το INT8
     μετακινεί τα σκορ κατά ~0.09 — αν στριμώξει το in-min κάτω από το κατώφλι,
     το σύστημα αρχίζει να ΑΡΝΕΙΤΑΙ σωστές απαντήσεις σιωπηλά. Το script
     ξαναϋπολογίζει και τα δύο άκρα με τα κβαντισμένα σκορ.
  3. Ταχύτητα.

Το μοντέλο κβαντίζεται σε ΑΝΤΙΓΡΑΦΟ (deepcopy): το ai_core.reranker είναι
singleton που χρησιμοποιεί το API — δεν το πειράζουμε ποτέ σε πείραμα.

    docker compose exec backend python evaluation/eval_int8_reranker.py
"""
import asyncio
import copy
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_set_50.jsonl")
REPEATS = 3


def load_tests() -> list:
    out = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def build_pairs(tests: list) -> list:
    """[(test, pairs)] — τα ΠΡΑΓΜΑΤΙΚΑ ζεύγη που φτάνουν στον reranker."""
    where = ai_core._build_where(None, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    out = []
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(dm, query, allowed, ai_core.DENSE_CANDIDATES)
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60, top_n=ai_core.RERANK_CANDIDATES)
        out.append((t, [[query, item[1]] for item in rrf]))
    return out


def score_all(model, batches: list) -> list:
    bs = ai_core.RERANK_BATCH_SIZE
    return [np.asarray(model.predict(p, batch_size=bs), dtype=np.float64)
            for _t, p in batches]


def time_all(model, batches: list) -> float:
    bs = ai_core.RERANK_BATCH_SIZE
    model.predict(batches[0][1], batch_size=bs)          # warmup
    per_query = []
    for _ in range(REPEATS):
        for _t, p in batches:
            t0 = time.perf_counter()
            model.predict(p, batch_size=bs)
            per_query.append((time.perf_counter() - t0) * 1000)
    return statistics.median(per_query)


def gate_margins(tests: list, scores: list) -> tuple:
    """(χειρότερο in-corpus best-score, καλύτερο out-of-corpus best-score).
    Ακριβώς τα δύο νούμερα πάνω στα οποία βαθμονομήθηκε το MIN_RERANK_SCORE."""
    in_best, out_best = [], []
    for (t, _pairs), s in zip(tests, scores):
        best = float(np.max(s))
        (out_best if t.get("category") == "out_of_corpus" else in_best).append(best)
    return min(in_best), max(out_best)


def main() -> int:
    tests = load_tests()
    batches = asyncio.run(build_pairs(tests))
    print(f"Ερωτήσεις: {len(batches)} (όλο το golden set, μαζί με out_of_corpus)")
    print(f"Μοντέλο: {ai_core.RERANKER_MODEL} · threads {torch.get_num_threads()} "
          f"· batch_size {ai_core.RERANK_BATCH_SIZE}\n")

    fp32 = ai_core.reranker
    s_fp32 = score_all(fp32, batches)
    ms_fp32 = time_all(fp32, batches)

    # ΑΝΤΙΓΡΑΦΟ + inplace: το quantize_dynamic ΧΩΡΙΣ inplace επιστρέφει νέο
    # module, και το CrossEncoder.forward του sentence-transformers 5.x περνά το
    # feature-dict ως POSITIONAL argument στα children -> σπάει. Με inplace
    # αντικαθίστανται μόνο τα Linear μέσα στο ίδιο αντικείμενο.
    qcross = copy.deepcopy(fp32)
    torch.ao.quantization.quantize_dynamic(
        qcross.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    s_int8 = score_all(qcross, batches)
    ms_int8 = time_all(qcross, batches)

    flat32, flat8 = np.concatenate(s_fp32), np.concatenate(s_int8)
    rho = float(np.corrcoef(flat32, flat8)[0, 1])
    flips = sum(1 for a, b in zip(s_fp32, s_int8)
                if int(np.argmax(a)) != int(np.argmax(b)))
    maxdiff = float(np.abs(flat32 - flat8).max())

    print(f"{'':<22}{'fp32':>12}{'int8':>12}")
    print("-" * 46)
    print(f"{'ms / ερώτηση':<22}{ms_fp32:>12.1f}{ms_int8:>12.1f}")
    print(f"{'επιτάχυνση':<22}{'1.00x':>12}{ms_fp32 / ms_int8:>11.2f}x")

    in_min32, out_max32 = gate_margins(batches, s_fp32)
    in_min8, out_max8 = gate_margins(batches, s_int8)
    gate = ai_core.MIN_RERANK_SCORE
    print(f"{'in-corpus χειρότερο':<22}{in_min32:>12.3f}{in_min8:>12.3f}")
    print(f"{'out-of-corpus καλύτ.':<22}{out_max32:>12.3f}{out_max8:>12.3f}")
    print(f"\nPearson ρ           : {rho:.4f}")
    print(f"max |διαφορά| σκορ  : {maxdiff:.4f}")
    print(f"top-1 flips         : {flips} / {len(batches)}")

    print(f"\n--- GATE (κατώφλι {gate}) ---")
    for label, in_min, out_max in (("fp32", in_min32, out_max32),
                                   ("int8", in_min8, out_max8)):
        head = in_min - gate          # πόσο πάνω από το κατώφλι το χειρότερο σωστό
        tail = gate - out_max         # πόσο κάτω το καλύτερο άσχετο
        status = "OK" if head > 0 and tail > 0 else "ΣΠΑΕΙ"
        print(f"  {label}: περιθώριο in-corpus +{head:.3f} · "
              f"out-of-corpus +{tail:.3f}  [{status}]")

    print("\nΚΡΙΤΗΡΙΟ: flips == 0 ΚΑΙ το περιθώριο in-corpus να μη μικρύνει "
          "κάτω από ~0.15.")
    if flips == 0 and (in_min8 - gate) > 0.15 and (gate - out_max8) > 0:
        print(f"-> ΠΕΡΝΑΕΙ. Κέρδος {ms_fp32 / ms_int8:.2f}x. "
              f"Επόμενο: πλήρες run_eval --retrieval-only για επιβεβαίωση.")
    else:
        print("-> ΑΠΟΡΡΙΠΤΕΤΑΙ ή θέλει επαναβαθμονόμηση του MIN_RERANK_SCORE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
