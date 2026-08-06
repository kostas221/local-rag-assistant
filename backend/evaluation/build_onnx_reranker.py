"""Export + INT8 quantization του cross-encoder, και ΜΕΤΡΗΣΗ πριν την υιοθέτηση.

ΓΙΑΤΙ ΤΩΡΑ, ΑΦΟΥ ΤΟ INT8 ΕΙΧΕ ΑΠΟΡΡΙΦΘΕΙ:
Το INT8 απορρίφθηκε για το bge-reranker-v2-m3 (568M, πολυγλωσσικό) — κατέστρεφε
την κατάταξη (top-1 overlap 0.625). Η αιτία, όπως τεκμηριώθηκε: το dynamic INT8
κόβει την ακρίβεια των cross-lingual attention heads, που χρειάζονται λεπτές
διαφορές floating point για να ξεχωρίσουν σημασίες μεταξύ γλωσσών. Τα DISTILLED
ΜΟΝΟΓΛΩΣΣΑ μοντέλα δεν έχουν αυτή την ευαισθησία. Τώρα ο reranker είναι
ms-marco-MiniLM-L-6-v2 (22M, αγγλικό) -> το INT8 αξίζει νέα μέτρηση.

ΚΙΝΗΤΡΟ: μετά τη μείωση από 15.048ms σε 693ms, ο reranker είναι ΠΑΛΙ το
bottleneck — 589ms από τα 611ms του warm retrieval (96%).

ΤΙ ΜΕΤΡΑΜΕ (τίποτα δεν αλλάζει στον παραγωγικό κώδικα — read only):
  1. ΠΟΙΟΤΗΤΑ: θέση του σωστού chunk, top-1 overlap, Spearman έναντι PyTorch
  2. LATENCY: PyTorch vs ONNX fp32 vs ONNX INT8, στη ΔΙΚΗ μας CPU
  3. GATE: το κατώφλι είναι δεμένο στην κλίμακα σκορ — αν το INT8 τη μετακινεί,
     το MIN_RERANK_SCORE πρέπει να ξαναοριστεί

    python evaluation/build_onnx_reranker.py            # export + μέτρηση
    python evaluation/build_onnx_reranker.py --rebuild  # ξανακάνει export
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

OUT_DIR = "/app/vector_db/onnx"
FP32 = os.path.join(OUT_DIR, "reranker_fp32.onnx")
INT8 = os.path.join(OUT_DIR, "reranker_int8.onnx")


def export_and_quantize(rebuild: bool = False) -> None:
    if os.path.exists(INT8) and not rebuild:
        print(f"Υπάρχει ήδη: {INT8} ({os.path.getsize(INT8)/1e6:.1f} MB)")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    model = ai_core.reranker.model.eval()
    tok = ai_core.reranker.tokenizer

    # Dummy ζεύγος (query, passage) — το BERT-based MiniLM θέλει και token_type_ids
    enc = tok(["what is cloud computing"], ["Cloud computing refers to services."],
              return_tensors="pt", padding=True, truncation=True, max_length=512)
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    args = tuple(enc[k] for k in names)

    print(f"Export -> ONNX fp32 (inputs: {names}) ...", flush=True)
    # dynamo=False: ΥΠΟΧΡΕΩΤΙΚΟ. Ο νέος (dynamo) exporter του torch 2.13 γράφει
    # τα weights σε ΞΕΧΩΡΙΣΤΟ .onnx.data (external data format) — 58KB graph +
    # 91MB data. Το quantize_dynamic του onnxruntime δεν το χειρίζεται και σκάει
    # στο save_and_reload_model_with_shape_infer. Ο legacy TorchScript exporter
    # γράφει ΕΝΙΑΙΟ αρχείο για μοντέλα <2GB, που είναι ό,τι θέλει το quantizer.
    torch.onnx.export(
        model, args, FP32,
        input_names=names, output_names=["logits"],
        # dynamic axes: batch ΚΑΙ μήκος ακολουθίας αλλάζουν σε κάθε κλήση
        dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"  fp32: {os.path.getsize(FP32)/1e6:.1f} MB")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    print("Quantization -> INT8 (dynamic, weights only) ...", flush=True)
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
    print(f"  int8: {os.path.getsize(INT8)/1e6:.1f} MB")


class OnnxReranker:
    """Ίδια διεπαφή με το CrossEncoder.predict -> drop-in στο ai_core."""

    def __init__(self, path: str, tokenizer, threads: int = 0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts,
                                         providers=["CPUExecutionProvider"])
        self.tok = tokenizer
        self.names = {i.name for i in self.sess.get_inputs()}

    def predict(self, pairs, batch_size: int = 16):
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            enc = self.tok([p[0] for p in chunk], [p[1] for p in chunk],
                           padding=True, truncation=True, max_length=512,
                           return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items()
                    if k in self.names}
            logits = self.sess.run(None, feed)[0]
            out.extend(float(x) for x in logits.reshape(-1))
        return out


def spearman(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: -a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: -b[i]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n * n - 1))


def first_hit(scored, kws):
    for r, (_s, text) in enumerate(sorted(scored, key=lambda x: -x[0]), 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    export_and_quantize(args.rebuild)
    tok = ai_core.reranker.tokenizer
    engines = {"torch": ai_core.reranker,
               "onnx_fp32": OnnxReranker(FP32, tok),
               "onnx_int8": OnnxReranker(INT8, tok)}

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    tests = []
    with open("/app/evaluation/golden_set_50.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    if args.limit:
        tests = tests[:args.limit]

    res = {k: {"ms": [], "hit": [], "best": [], "rho": [], "top1": 0}
           for k in engines}
    print(f"\n{'id':<7} {'cat':<13} " +
          " ".join(f"{k:>10}" for k in engines) + "   (θέση σωστού chunk)")
    print("-" * 74)

    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(dm, query, allowed,
                                         ai_core.DENSE_CANDIDATES)
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed,
                                         ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], top_n=ai_core.RERANK_CANDIDATES)
        pairs = [[query, it[1]] for it in rrf]
        texts = [it[1] for it in rrf]

        scores, base_top = {}, None
        for name, eng in engines.items():
            t0 = time.perf_counter()
            sc = [float(x) for x in eng.predict(pairs)]
            res[name]["ms"].append((time.perf_counter() - t0) * 1000)
            scores[name] = sc
            ordered = sorted(zip(sc, texts), key=lambda x: -x[0])
            if name == "torch":
                base_top = ordered[0][1]
            elif ordered[0][1] == base_top:
                res[name]["top1"] += 1
            res[name]["hit"].append(first_hit(list(zip(sc, texts)),
                                              t["keywords"]))
            res[name]["best"].append((max(sc), t.get("category")))
            res[name]["rho"].append(spearman(scores["torch"], sc))

        print(f"{t['id']:<7} {t.get('category',''):<13} " +
              " ".join(f"{res[k]['hit'][-1] or '-'!s:>10}" for k in engines))

    print("\n" + "=" * 74)
    print("ΠΟΙΟΤΗΤΑ ΚΑΤΑΤΑΞΗΣ (αναφορά: torch)")
    print("=" * 74)
    inc = [i for i, t in enumerate(tests) if t.get("category") != "out_of_corpus"]
    for k in engines:
        hits = [res[k]["hit"][i] for i in inc if res[k]["hit"][i]]
        lost = [tests[i]["id"] for i in inc
                if res["torch"]["hit"][i] and not res[k]["hit"][i]]
        print(f"  {k:<10} διάμεσο rank {statistics.median(hits):>4.1f} | "
              f"top-1 συμφωνία {res[k]['top1']:>2}/{len(tests)} | "
              f"διάμεσο Spearman {statistics.median(res[k]['rho']):>5.2f}"
              + (f" | *** ΧΑΘΗΚΑΝ: {lost} ***" if lost else ""))

    print("\n" + "=" * 74)
    print(f"LATENCY ({ai_core.RERANK_CANDIDATES} υποψήφια, αυτή η CPU)")
    print("=" * 74)
    base = statistics.median(res["torch"]["ms"])
    for k in engines:
        m = statistics.median(res[k]["ms"])
        print(f"  {k:<10} {m:>8.0f} ms" +
              (f"   -> {base/m:.1f}x ταχύτερο" if k != "torch" else "   (αναφορά)"))

    print("\n" + "=" * 74)
    print("GATE — η κλίμακα σκορ ΑΛΛΑΖΕΙ ανά engine")
    print("=" * 74)
    for k in engines:
        i_lo = min(s for s, c in res[k]["best"] if c != "out_of_corpus")
        ooc = [s for s, c in res[k]["best"] if c == "out_of_corpus"]
        o_hi = max(ooc) if ooc else float("nan")
        gap = i_lo - o_hi
        print(f"  {k:<10} in-min {i_lo:>8.3f} | out-max {o_hi:>8.3f} | "
              f"διάκενο {gap:>+7.3f} "
              f"{'ΚΑΘΑΡΟ' if gap > 0 else '*** ΕΠΙΚΑΛΥΨΗ ***'}")
        if gap > 0:
            # Ασύμμετρα υπέρ της άρνησης: μια λάθος απάντηση σε ερώτηση εκτός
            # corpus κοστίζει περισσότερο από μια χαμένη δύσκολη ανάκτηση.
            print(f"  {'':<10} προτεινόμενο MIN_RERANK_SCORE: "
                  f"{o_hi + gap * 0.7:.2f}")
    return 0


sys.exit(asyncio.run(main()))
