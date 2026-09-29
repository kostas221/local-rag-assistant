"""Φάση 1.2, βήμα 1: Qwen3-Embedding-0.6B έναντι bge-m3 — ΤΑΧΥΤΗΤΑ + πρώτη ένδειξη, 0 Gemini.

ΓΙΑΤΙ ΠΡΩΤΑ ΑΥΤΟ: ίδια λογική με το bench_reranker_latency.py — αν το μοντέλο δεν χωράει στον
χρόνο, όλη η υπόλοιπη δουλειά (νέο store, scoreboard, κριτής) είναι χαμένη. Εδώ ο κίνδυνος είναι
μικρότερος (ίδιο μέγεθος: ~600M και τα δύο), αλλά ΔΕΝ υποτίθεται — μετριέται.

ΠΟΥ ΚΟΣΤΙΖΕΙ ΤΟ EMBEDDING ΣΤΟ ΣΥΣΤΗΜΑ (γι' αυτό μετράμε δύο πράγματα, όχι ένα):
  (α) ΕΡΩΤΗΣΗ: μία φορά ανά ΝΕΑ ερώτηση (μετά μένει στο cache) -> το «κρύο» μονοπάτι του χρήστη
  (β) INGEST: κάθε κομμάτι κάθε PDF που ανεβαίνει -> ο χρόνος ανεβάσματος
Το ζεστό μονοπάτι (ερώτηση ήδη στο cache, πίνακας στη μνήμη) ΔΕΝ αλλάζει: matmul 418×1024 και στα δύο.

ΤΙ ΚΑΝΕΙ:
  1. Ανοίγει το απομονωμένο store (418 chunks, διανύσματα bge-m3) — ΚΑΝΕΝΑ άγγιγμα στην παραγωγή.
     Το cache διανυσμάτων ερωτήσεων της παραγωγής ΔΕΝ χρησιμοποιείται ούτε γράφεται: το Qwen3 βγάζει
     ΕΠΙΣΗΣ 1024 διαστάσεις, άρα ο έλεγχος διάστασης του cache ΔΕΝ θα έπιανε ανάμειξη.
  2. bge-m3: ξαναφτιάχνει 100 κομμάτια (χρόνος + ΕΛΕΓΧΟΣ: ίδια με τα αποθηκευμένα, cos ≈ 1.0).
  3. Qwen3 (float32 — στο CPU το bf16 είναι αργό/ανακριβές): και τα 418 κομμάτια.
  4. Ερωτήσεις 5 σετ (κύριο, 2 multi_hop, 2 δύσκολα — ίδιες μεταφράσεις με το baseline.json):
     χρόνος ανά ερώτηση + ΜΟΝΟ dense ανάκτηση top-30: κάλυψη λέξεων, MRR ανά λέξη, «και τα δύο».
     Qwen3 ΜΕ την οδηγία ερώτησης του μοντέλου (prompt_name="query") ΚΑΙ ΧΩΡΙΣ — το σημερινό
     embedding function της Chroma δεν μπορεί να βάλει οδηγία, άρα μετράμε αν χρειάζεται.

⚠️ Το dense top-30 είναι ΕΝΔΙΑΜΕΣΟ στάδιο: ακολουθούν BM25, RRF και reranker, που αναδιατάσσουν.
Η απόφαση για ποιότητα παίρνεται ΜΟΝΟ από το scoreboard (βήμα 2), όχι από εδώ.

═══ ΠΡΟΒΛΕΨΗ — γραμμένη ΠΡΙΝ (29/9/2026) ═══
  ερώτηση (1 κείμενο)    bge-m3 50-200 ms · Qwen3 ±30% του bge-m3
  ingest ανά 100 chunks  Qwen3 1.0-1.5× του bge-m3
  έλεγχος bge-m3         cos ≥ 0.999 σε 100/100
  dense top-30 κάλυψη    Qwen3 (με οδηγία) από −2 έως +6 μονάδες % έναντι bge-m3 — όχι ξεκάθαρο
  οδηγία                 με > χωρίς, κατά 0-3 μονάδες

═══ ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ — γραμμένος ΠΡΙΝ ═══
  ερώτηση ≤ 1.25× bge-m3 ΚΑΙ ingest ≤ 2× bge-m3  -> βήμα 2 (store με Qwen3 + scoreboard)
  αλλιώς                                          -> σταματάμε και αποφασίζουμε μαζί
  Το dense top-30 ΔΕΝ μπαίνει στον κανόνα (ενδιάμεσο στάδιο).

═══ ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/embedder_bench_Qwen3-Embedding-0.6B.txt) ═══
  ερώτηση (διάμεσος, 160)   bge-m3 184 ms · Qwen3 253 ms -> 1.38×   ✗ ΚΑΝΟΝΑΣ (≤1.25×)
  ingest ανά 100 chunks     bge-m3 83.5 s · Qwen3 153.6 s -> 1.84×  ✓ κανόνας (≤2×)
                            (418 chunks: ~6 -> ~11 λεπτά· ένα paper 20 σελίδων ~1 -> ~2 λεπτά)
  έλεγχος bge-m3            cos 1.00000 σε 100/100 ✓ (το store = bge-m3 του ίδιου κειμένου)
  -> Ο κανόνας λέει «σταματάμε και αποφασίζουμε μαζί». ΣΕ ΑΠΟΛΥΤΑ: +69 ms ΜΟΝΟ σε ΝΕΑ ερώτηση
     (μετά cache)· το ζεστό μονοπάτι ΙΔΙΟ. Κρύα ανάκτηση ~0.77+0.18 = 0.95 s -> ~1.02 s, ΜΕΣΑ στον
     προϋπολογισμό της Φάσης 1 (≤1.2 s, 25/9). Το 1.25× ήταν δικό μου υποκατάστατο, αυστηρότερο.
  dense top-30 (ενδιάμεσο):  κάλυψη 82.6% -> 84.7% (+οδηγία) · MRR 0.375 -> 0.433 · ανά ερώτηση
     καλύτερα 19 / χειρότερα 9 · multi_hop νέες κάλυψη 67.5 -> 77.6%, «και τα δύο» 12 -> 17/29 ·
     hard_old MRR 0.235 -> 0.327. Οδηγία: ΟΛΑ 84.7 vs 81.9 χωρίς (αλλά mh_new 77.6 vs 79.3).
  Πρόβλεψη: ερώτηση ±30% ✗ (1.38) · ingest 1.0-1.5× ✗ (1.84) · έλεγχος ✓ · κάλυψη −2..+6 ✓ (+2.1)
     · οδηγία 0-3 ✓ (+2.8). ΥΠΟΤΙΜΗΣΑ ΤΟ ΚΟΣΤΟΣ (το Qwen3 έχει 28 στρώματα έναντι 24).

    docker compose exec backend python evaluation/bench_embedder.py
Την 1η φορά κατεβάζει το Qwen3-Embedding-0.6B (~1.2 GB). ~8-12 λεπτά. ΜΗΝ τρέχει μαζί με άλλο eval.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
BASELINE = os.path.join(RUNS, "scoreboard", "baseline.json")
LOCK = "/tmp/scoreboard.lock"  # noqa: S108  ΙΔΙΑ κλειδαριά με το scoreboard (κοινό store)
SETS = [("main", "golden_set_50.jsonl"), ("mh_old", "golden_multihop_new.jsonl"),
        ("mh_new", "golden_multihop_v2.jsonl"), ("hard_old", "golden_hard_paraphrase.jsonl"),
        ("hard_new", "golden_hard_new.jsonl")]
TOP = 30                                   # = DENSE_CANDIDATES της παραγωγής


class _Tee:
    """Ό,τι τυπώνεται πάει ΚΑΙ σε αρχείο στο runs/."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")  # noqa: SIM115  ζει όσο το script
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def norm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def load_questions() -> list[dict]:
    """Ερωτήσεις με ΙΔΙΑ μετάφραση όπως στο baseline.json (0 Gemini) + λέξεις-κλειδιά."""
    with open(BASELINE, encoding="utf-8") as f:
        base = {r["key"]: r for r in json.load(f)["rows"]}
    out = []
    for key, fname in SETS:
        for t in E.load_jsonl(os.path.join(HERE, fname)):
            if t.get("category") == "out_of_corpus":
                continue
            r = base.get(f"{key}:{t['id']}")
            if r is None:
                continue
            out.append({"set": key, "id": t["id"], "query": r["translation"] or r["question"],
                        "kw": t.get("keywords") or [], "by_doc": t.get("keywords_by_doc") or {}})
    return out


def dense_scores(qv: np.ndarray, M: np.ndarray, docs, metas, q: dict) -> dict:
    order = np.argsort(-(M @ qv), kind="stable")[:TOP]
    top = [(docs[i].lower(), metas[i].get("file_name")) for i in order]
    ranks = []
    for k in q["kw"]:
        r = next((j + 1 for j, (d, _f) in enumerate(top) if k.lower() in d), None)
        ranks.append(r)
    cov = sum(r is not None for r in ranks) / len(ranks) if ranks else 0.0
    mrr = statistics.mean(1 / r if r else 0.0 for r in ranks) if ranks else 0.0
    both = None
    if q["by_doc"]:
        both = int(all(any(k.lower() in d for k in kws for d, f in top if f == doc)
                       for doc, kws in q["by_doc"].items()))
    return {"cov": cov, "mrr": mrr, "both": both}


def run(args) -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    ai_core, _saved = E.open_store()
    if ai_core.collection.count() != 418:
        print(f"!! Το store έχει {ai_core.collection.count()} chunks, όχι 418 — σταματάω.")
        return 1
    data = ai_core.collection.get(include=["documents", "embeddings", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    M_bge = norm(np.asarray(data["embeddings"], dtype=np.float32))
    bge = ai_core.sentence_transformer_ef._model
    print(f"torch threads {torch.get_num_threads()} · chunks {len(docs)} · "
          f"bge-m3 dtype {next(bge.parameters()).dtype}")

    # ---- 2: bge-m3, 100 κομμάτια: χρόνος + έλεγχος ακεραιότητας ----
    n_chk = min(100, len(docs))
    bge.encode(docs[:4], batch_size=32)                       # ζέσταμα
    t0 = time.perf_counter()
    V = bge.encode(docs[:n_chk], batch_size=32, normalize_embeddings=True)
    t_bge100 = time.perf_counter() - t0
    cos = np.sum(V * M_bge[:n_chk], axis=1)
    print(f"\nbge-m3: {n_chk} κομμάτια σε {t_bge100:.1f} s · έλεγχος cos με το store: "
          f"min {cos.min():.5f} · ≥0.999 σε {int((cos >= 0.999).sum())}/{n_chk}")

    # ---- 3: Qwen3, όλα τα κομμάτια ----
    print(f"\nΦόρτωση {args.model} (την 1η φορά κατεβαίνει ~1.2 GB) ...", flush=True)
    t0 = time.perf_counter()
    qwen = SentenceTransformer(args.model, device=ai_core.DEVICE,
                               model_kwargs={"dtype": torch.float32})
    print(f"  φορτώθηκε σε {time.perf_counter() - t0:.1f} s · dtype "
          f"{next(qwen.parameters()).dtype} · διάσταση {qwen.encode(['x']).shape[-1]} · "
          f"οδηγίες {list((qwen.prompts or {}).keys())}")
    qwen.encode(docs[:4], batch_size=32)                      # ζέσταμα
    t0 = time.perf_counter()
    M_qw = np.asarray(qwen.encode(docs, batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=False), dtype=np.float32)
    t_qw_all = time.perf_counter() - t0
    t_qw100 = t_qw_all * n_chk / len(docs)
    print(f"Qwen3: {len(docs)} κομμάτια σε {t_qw_all:.1f} s (ανά {n_chk}: {t_qw100:.1f} s)")

    # ---- 4: ερωτήσεις: χρόνος + dense top-30 ----
    qs = load_questions()
    per_set = ", ".join(f"{k} {sum(q['set'] == k for q in qs)}" for k, _ in SETS)
    print(f"\nΕρωτήσεις: {len(qs)} ({per_set})")
    variants = {
        "bge-m3": lambda q: bge.encode([q], normalize_embeddings=True)[0],
        "qwen3+οδηγία": lambda q: qwen.encode([q], prompt_name="query", normalize_embeddings=True)[0],
        "qwen3 χωρίς": lambda q: qwen.encode([q], normalize_embeddings=True)[0],
    }
    mats = {"bge-m3": M_bge, "qwen3+οδηγία": M_qw, "qwen3 χωρίς": M_qw}
    for f in variants.values():                               # ζέσταμα
        f("warm up")
    times = {v: [] for v in variants}
    res = {v: [] for v in variants}
    for q in qs:
        for v, f in variants.items():                         # εναλλάξ: ίδιος θόρυβος μηχανήματος
            t0 = time.perf_counter()
            qv = np.asarray(f(q["query"]), dtype=np.float32)
            times[v].append(time.perf_counter() - t0)
            res[v].append(dict(dense_scores(qv, mats[v], docs, metas, q), set=q["set"], id=q["id"]))

    # ---- αναφορά ----
    print("\n" + "=" * 86)
    print("ΤΑΧΥΤΗΤΑ")
    print("=" * 86)
    tb = statistics.median(times["bge-m3"])
    tq = statistics.median(times["qwen3+οδηγία"])
    print(f"  ερώτηση (διάμεσος)   bge-m3 {tb * 1000:.0f} ms · Qwen3 {tq * 1000:.0f} ms  -> {tq / tb:.2f}×")
    print(f"  ingest ανά {n_chk}      bge-m3 {t_bge100:.1f} s · Qwen3 {t_qw100:.1f} s  -> "
          f"{t_qw100 / t_bge100:.2f}×   (418 κομμάτια: bge-m3 ~{t_bge100 * len(docs) / n_chk:.0f} s · "
          f"Qwen3 {t_qw_all:.0f} s)")

    print("\n" + "=" * 86)
    print(f"ΜΟΝΟ DENSE, top-{TOP} (ενδιάμεσο στάδιο — ΔΕΝ αποφασίζει)")
    print("=" * 86)
    print(f"  {'σετ':<10}{'n':>4}" + "".join(f"{v + ' κάλ.':>18}{'MRR':>7}" for v in variants))
    for key in [k for k, _ in SETS] + ["ΟΛΑ"]:
        line = None
        for v in variants:
            rs = [r for r in res[v] if key == "ΟΛΑ" or r["set"] == key]
            if line is None:
                line = f"  {key:<10}{len(rs):>4}"
            line += (f"{100 * statistics.mean(r['cov'] for r in rs):>17.1f}%"
                     f"{statistics.mean(r['mrr'] for r in rs):>7.3f}")
        print(line)
    for v in variants:
        b = [r["both"] for r in res[v] if r["both"] is not None]
        print(f"  multi_hop νέες «και τα δύο papers» στο top-{TOP}: {v:<14} {sum(b)}/{len(b)}")
    better = sum(q["cov"] > b["cov"] for q, b in zip(res["qwen3+οδηγία"], res["bge-m3"], strict=True))
    worse = sum(q["cov"] < b["cov"] for q, b in zip(res["qwen3+οδηγία"], res["bge-m3"], strict=True))
    print(f"  ανά ερώτηση (κάλυψη), Qwen3+οδηγία έναντι bge-m3: καλύτερα {better} · χειρότερα {worse}")

    ok = tq <= 1.25 * tb and t_qw100 <= 2 * t_bge100
    print(f"\n  ΕΤΥΜΗΓΟΡΙΑ ΤΑΧΥΤΗΤΑΣ (κανόνας γραμμένος πριν): "
          f"{'ΣΥΝΕΧΙΖΟΥΜΕ στο βήμα 2' if ok else 'ΣΤΑΜΑΤΑΜΕ — αποφασίζουμε μαζί'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    args = ap.parse_args()
    out = os.path.join(RUNS, f"embedder_bench_{args.model.split('/')[-1]}.txt")
    sys.stdout = _Tee(out)
    print(f"(η έξοδος σώζεται και στο {os.path.relpath(out, HERE)})")
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard/eval στο κοινό store (ή έμεινε κλειδαριά: σβήσε το {LOCK}).")
        return 1
    try:
        return run(args)
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
