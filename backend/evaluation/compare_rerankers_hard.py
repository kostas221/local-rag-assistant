"""Είναι ο reranker η ΡΙΖΑ του προβλήματος; — σύγκριση στο hard set.

ΓΙΑΤΙ ΞΑΝΑ (το compare_rerankers.py έτρεξε ήδη το 2025):
Το bge-reranker-v2-m3 απορρίφθηκε με μέτρηση ΑΠΟΚΛΕΙΣΤΙΚΑ στο golden_set_50 —
το σετ που αποδείχθηκε ότι έχει survivorship bias (το gate έβγαζε 61/61 επειδή
είχαμε πετάξει την ερώτηση που το χαλούσε). Στο hard set το ΤΡΕΧΟΝ μοντέλο
κόβει 11/16 σωστές ερωτήσεις. Η απόρριψη πρέπει να ξαναελεγχθεί εκεί που πονάει.

Η ΥΠΟΘΕΣΗ: το MiniLM-L-6 (22M, αγγλικό) είναι LEXICAL — καταρρέει όταν το
ερώτημα δεν μοιράζεται λέξεις με το κείμενο. Το bge-v2-m3 (568M, πολυγλωσσικό)
θα έπρεπε να αντέχει καλύτερα. Αν ΔΕΝ αντέχει, η αδυναμία είναι ΔΟΜΙΚΗ του
cross-encoder reranking — και ο corrective agent είναι η σωστή λύση, όχι επίδεσμος.

ΤΙ ΚΑΝΕΙ ΣΩΣΤΑ (και το κάνει συγκρίσιμο):
  - ΙΔΙΑ υποψήφια για όλα τα μοντέλα: ένα RRF pass, μετά μόνο ο reranker αλλάζει.
    Χωρίς αυτό συγκρίνονται δύο διαφορετικά pipelines, όχι δύο rerankers.
  - ΑΝΕΞΑΡΤΗΤΟ ΑΠΟ ΚΛΙΜΑΚΑ: το bge δίνει sigmoid [0,1], το MiniLM ωμά logits
    [-11,+11]. Σύγκριση σκορ θα ήταν ανοησία. Αντ' αυτού, ανά μοντέλο:
      thr* = max(out_of_corpus) + ε  -> το ΕΛΑΧΙΣΤΟ κατώφλι που κρατά 5/5
      μετράμε πόσες in-corpus/hard περνάνε ΜΕ ΑΥΤΟ.
    Δηλαδή: «με ίδια εγγύηση μη-ψευδαίσθησης, ποιο σώζει περισσότερες;»
  - kw_rank: θέση του σωστού chunk. ΕΝΤΕΛΩΣ ανεξάρτητο από κατώφλι — αν ένα
    μοντέλο κατατάσσει σταθερά καλύτερα, φαίνεται εδώ χωρίς βαθμονόμηση.

ΚΟΣΤΟΣ: μηδέν API. Χρόνος: το bge είναι ~20x αργότερο -> ~8 λεπτά συνολικά.
RAM: +2.3GB για το bge πάνω από τα φορτωμένα (container 11.7GB, χωράει).

    docker compose exec backend python evaluation/compare_rerankers_hard.py
    docker compose exec backend python evaluation/compare_rerankers_hard.py --models cross-encoder/ms-marco-MiniLM-L-12-v2
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

HARD = "/app/evaluation/golden_hard_paraphrase.jsonl"
MAIN = "/app/evaluation/golden_set_50.jsonl"


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def first_hit(ordered, kws):
    for r, (_s, text) in enumerate(ordered, 1):
        if any(k.lower() in text.lower() for k in kws):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["BAAI/bge-reranker-v2-m3"],
                    help="υποψήφιοι πέρα από τον τρέχοντα (που μπαίνει πάντα)")
    ap.add_argument("--n-incorpus", type=int, default=10,
                    help="πόσες κανονικές in-corpus (έλεγχος μη-υποβάθμισης)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # Δείγμα: ΟΛΟ το hard set + ΟΛΑ τα out_of_corpus + n κανονικές.
    # Τα out_of_corpus είναι υποχρεωτικά ΟΛΑ — ορίζουν το κατώφλι κάθε μοντέλου.
    hard = load(HARD)
    main_set = load(MAIN)
    ooc = [t for t in main_set if t.get("category") == "out_of_corpus"]
    normal = [t for t in main_set
              if t.get("category") != "out_of_corpus"][:args.n_incorpus]
    tests = [("hard", t) for t in hard] + [("normal", t) for t in normal] \
        + [("ooc", t) for t in ooc]
    print(f"{len(hard)} hard · {len(normal)} κανονικές · {len(ooc)} out_of_corpus "
          f"= {len(tests)} ερωτήσεις\n")

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    # --- ΕΝΑ RRF pass: τα υποψήφια είναι ΚΟΙΝΑ για όλα τα μοντέλα ---
    print("Υποψήφια (ένα RRF pass, κοινά για όλα τα μοντέλα)...", flush=True)
    shared = []
    for group, t in tests:
        q = ai_core._translation_cache.get(t["question"], t["question"])
        d = ai_core._dense_exact_ids(dm, q, allowed,
                                     min(ai_core.DENSE_CANDIDATES, len(allowed)))
        s = ai_core._bm25_sparse_ids(idx, q, allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d, s, idx["ids"], idx["texts"], idx["metas"],
                                k=60, top_n=ai_core.RERANK_CANDIDATES,
                                pos=idx["pos"])
        shared.append((group, t, q, rrf))
    print(f"  {len(shared)} σύνολα υποψηφίων έτοιμα\n")

    from sentence_transformers import CrossEncoder
    models = {ai_core.RERANKER_MODEL: ai_core.reranker}
    for name in args.models:
        if name in models:
            continue
        print(f"Φόρτωση {name} ...", flush=True)
        t0 = time.time()
        models[name] = CrossEncoder(name, device=ai_core.DEVICE)
        print(f"  {time.time()-t0:.0f}s\n")

    rows = []
    for name, model in models.items():
        print(f"--- {name} ---", flush=True)
        t0 = time.time()
        for group, t, q, rrf in shared:
            pairs = [[q, it[1]] for it in rrf]
            sc = [float(x) for x in
                  model.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
            ordered = sorted(zip(sc, [it[1] for it in rrf]), key=lambda x: -x[0])
            rows.append(dict(
                model=name, group=group, id=t["id"],
                category=t.get("category", ""), best=ordered[0][0],
                kw_rank=(None if group == "ooc" else first_hit(ordered, t["keywords"]))))
        el = time.time() - t0
        print(f"  {el:.0f}s συνολικά · {1000*el/len(shared):.0f} ms/ερώτηση\n",
              flush=True)

    print("=" * 84)
    print("ΑΠΟΤΕΛΕΣΜΑ — «με ΙΔΙΑ εγγύηση out_of_corpus 5/5, ποιο σώζει περισσότερες;»")
    print("=" * 84)
    print(f"{'μοντέλο':<38} {'thr*':>8} {'hard':>8} {'κανον.':>8} {'kw@ διάμ.':>10}")
    print("-" * 84)

    summary = []
    for name in models:
        mine = [r for r in rows if r["model"] == name]
        ooc_s = [r["best"] for r in mine if r["group"] == "ooc"]
        # Το ΕΛΑΧΙΣΤΟ κατώφλι που κόβει ΟΛΑ τα out_of_corpus.
        thr = max(ooc_s) + 1e-6
        h = [r for r in mine if r["group"] == "hard"]
        n = [r for r in mine if r["group"] == "normal"]
        h_pass = sum(1 for r in h if r["best"] >= thr)
        n_pass = sum(1 for r in n if r["best"] >= thr)
        ranks = [r["kw_rank"] for r in h if r["kw_rank"]]
        med = statistics.median(ranks) if ranks else float("nan")
        found = len(ranks)
        summary.append((name, thr, h_pass, len(h), n_pass, len(n), med, found))
        print(f"{name:<38} {thr:>8.3f} {f'{h_pass}/{len(h)}':>8} "
              f"{f'{n_pass}/{len(n)}':>8} {med:>10.1f}")

    print("\n  thr* = το ΕΛΑΧΙΣΤΟ κατώφλι που κρατά out_of_corpus 5/5 στο ΙΔΙΟ μοντέλο.")
    print("  Τα σκορ ΔΕΝ συγκρίνονται μεταξύ μοντέλων (sigmoid vs ωμά logits) —")
    print("  συγκρίνονται μόνο τα ΠΟΣΟΣΤΑ επιτυχίας κάτω από ίδια εγγύηση.")

    print("\n  ΘΕΣΗ ΣΩΣΤΟΥ CHUNK στο hard set (ανεξάρτητο κατωφλίου):")
    for name, _thr, _hp, _ht, _np, _nt, med, found in summary:
        print(f"    {name:<38} βρέθηκε σε {found}/{len(hard)} · διάμεσος θέση {med:.1f}")

    if len(summary) > 1:
        base = summary[0]
        print("\n" + "=" * 84)
        print("ΕΤΥΜΗΓΟΡΙΑ")
        print("=" * 84)
        for cand in summary[1:]:
            dh = cand[2] - base[2]
            if dh > 1:
                print(f"  {cand[0]}: +{dh} hard ερωτήσεις -> Η ΡΙΖΑ ΕΙΝΑΙ Ο RERANKER. "
                      f"Επόμενο: ενδιάμεσα μοντέλα για το latency.")
            elif dh < -1:
                print(f"  {cand[0]}: {dh} hard -> ΧΕΙΡΟΤΕΡΟ. Η απόρριψη του 2025 "
                      f"επιβεβαιώνεται ΚΑΙ στο δύσκολο άκρο.")
            else:
                print(f"  {cand[0]}: {dh:+d} hard -> ΙΣΟΠΑΛΙΑ. Η αδυναμία είναι "
                      f"ΔΟΜΙΚΗ του cross-encoder reranking, όχι του μεγέθους. "
                      f"Ο corrective agent είναι η ΣΩΣΤΗ λύση, όχι επίδεσμος.")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
