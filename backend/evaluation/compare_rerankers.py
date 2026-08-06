"""Αντέχει ένα 22M αγγλικό cross-encoder στη θέση του 568M πολυγλωσσικού;

ΤΟ ΣΚΕΠΤΙΚΟ (από την έκθεση έρευνας, και επιβεβαιώνεται στον κώδικα):
το optimize_query() μεταφράζει τα ελληνικά σε αγγλικά ΠΡΙΝ την ανάκτηση, και το
corpus είναι αγγλικό. Άρα ο reranker βλέπει ΠΑΝΤΑ (αγγλικό ερώτημα, αγγλικό
chunk) — πληρώνουμε cross-attention για 100+ γλώσσες που δεν χρησιμοποιείται.

ΤΙ ΜΕΤΡΑΜΕ (τίποτα δεν αλλάζει στον κώδικα — read only):
  1. ΠΟΙΟΤΗΤΑ: συμφωνία κατάταξης στα ΙΔΙΑ 15 υποψήφια (top-1 overlap, Spearman)
     και — το μόνο που μετράει πραγματικά — σε ποια θέση βάζει ο καθένας το
     chunk που περιέχει τα σωστά keywords.
  2. LATENCY: πραγματικός χρόνος στη ΔΙΚΗ μας CPU, όχι εκτιμήσεις από paper.
  3. GATE: κατανομή σκορ in-corpus vs out-of-corpus. ΚΡΙΣΙΜΟ — το bge δίνει
     sigmoid [0,1] και το κατώφλι 0.05 είναι βαθμονομημένο γι' αυτό. Άλλο μοντέλο
     = άλλη κλίμακα (πιθανά logits [-11,+11]) -> το gate ΠΡΕΠΕΙ να ξαναοριστεί.

ΠΡΟΣΟΧΗ: ΜΗΝ το τρέχεις ταυτόχρονα με eval — φορτώνει δεύτερο μοντέλο και
ανταγωνίζεται για CPU/RAM.

    python _reranker_compare.py                    # default: MiniLM-L-6
    python _reranker_compare.py --model cross-encoder/ms-marco-MiniLM-L-12-v2
    python _reranker_compare.py --model BAAI/bge-reranker-base
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS


def spearman(a: list, b: list) -> float:
    """Συντελεστής Spearman χωρίς scipy (μικρά n, χωρίς ισοβαθμίες στα ranks)."""
    n = len(a)
    if n < 2:
        return 1.0
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: -a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: -b[i]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n * n - 1))


def first_hit(items, kws):
    """Θέση (1-based) του πρώτου chunk με keyword· None αν δεν υπάρχει."""
    for r, (_s, text) in enumerate(items, 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    print(f"Φόρτωση υποψήφιου: {args.model} ...", flush=True)
    t0 = time.time()
    cand = CrossEncoder(args.model, device=ai_core.DEVICE)
    print(f"  φορτώθηκε σε {time.time()-t0:.1f}s")
    act = getattr(cand, "activation_fn", None) or getattr(cand, "default_activation_function", None)
    print(f"  activation: {type(act).__name__ if act else '(κανένα -> ωμά logits)'}")
    n_params = sum(p.numel() for p in cand.model.parameters())
    base_params = sum(p.numel() for p in ai_core.reranker.model.parameters())
    print(f"  παράμετροι: {n_params/1e6:.1f}M  vs  bge {base_params/1e6:.1f}M "
          f"({base_params/n_params:.1f}x μικρότερο)\n")

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed_ids = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    tests = []
    with open("/app/evaluation/golden_set_50.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    if args.limit:
        tests = tests[:args.limit]

    print(f"{'id':<7} {'cat':<13} {'bge#':>5} {'new#':>5} {'top1':>5} "
          f"{'ρ':>6} {'bge_ms':>8} {'new_ms':>8} {'bge_best':>9} {'new_best':>9}")
    print("-" * 88)

    rows = []
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(dm, query, allowed_ids,
                                         ai_core.DENSE_CANDIDATES)
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed_ids,
                                         ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60, top_n=ai_core.RERANK_CANDIDATES)
        pairs = [[query, it[1]] for it in rrf]
        texts = [it[1] for it in rrf]

        t0 = time.time()
        s_bge = [float(x) for x in ai_core.reranker.predict(pairs)]
        ms_bge = (time.time() - t0) * 1000

        t0 = time.time()
        s_new = [float(x) for x in cand.predict(pairs)]
        ms_new = (time.time() - t0) * 1000

        ord_bge = sorted(zip(s_bge, texts), key=lambda x: -x[0])
        ord_new = sorted(zip(s_new, texts), key=lambda x: -x[0])
        top1 = "ναι" if ord_bge[0][1] == ord_new[0][1] else "ΟΧΙ"
        rho = spearman(s_bge, s_new)

        kws = t["keywords"]
        h_bge = first_hit(ord_bge, kws)
        h_new = first_hit(ord_new, kws)

        rows.append(dict(id=t["id"], cat=t.get("category", ""), rho=rho,
                         top1=(top1 == "ναι"), ms_bge=ms_bge, ms_new=ms_new,
                         h_bge=h_bge, h_new=h_new,
                         best_bge=max(s_bge), best_new=max(s_new)))
        print(f"{t['id']:<7} {t.get('category',''):<13} "
              f"{h_bge or '-'!s:>5} {h_new or '-'!s:>5} {top1:>5} "
              f"{rho:>6.2f} {ms_bge:>8.0f} {ms_new:>8.0f} "
              f"{max(s_bge):>9.3f} {max(s_new):>9.3f}", flush=True)

    inc = [r for r in rows if r["cat"] != "out_of_corpus"]
    ooc = [r for r in rows if r["cat"] == "out_of_corpus"]

    print("\n" + "=" * 88)
    print("ΠΟΙΟΤΗΤΑ ΚΑΤΑΤΑΞΗΣ")
    print("=" * 88)
    print(f"  top-1 συμφωνία: {sum(r['top1'] for r in rows)}/{len(rows)}")
    print(f"  διάμεσο Spearman: {statistics.median(r['rho'] for r in rows):.3f}")
    both = [(r["h_bge"], r["h_new"]) for r in inc if r["h_bge"] and r["h_new"]]
    if both:
        print(f"  θέση σωστού chunk — bge διάμεσος {statistics.median(a for a,_ in both):.1f}"
              f" | νέο {statistics.median(b for _,b in both):.1f}")
        better = sum(1 for a, b in both if b < a)
        worse = sum(1 for a, b in both if b > a)
        print(f"  νέο καλύτερο σε {better}, χειρότερο σε {worse}, "
              f"ίδιο σε {len(both)-better-worse}")
    lost = [r["id"] for r in inc if r["h_bge"] and not r["h_new"]]
    gained = [r["id"] for r in inc if r["h_new"] and not r["h_bge"]]
    if lost:
        print(f"  *** ΧΑΘΗΚΑΝ εντελώς με το νέο: {lost} ***")
    if gained:
        print(f"  βρέθηκαν μόνο με το νέο: {gained}")

    print("\n" + "=" * 88)
    print("LATENCY (15 υποψήφια, αυτή η CPU)")
    print("=" * 88)
    mb = statistics.median(r["ms_bge"] for r in rows)
    mn = statistics.median(r["ms_new"] for r in rows)
    print(f"  bge-reranker-v2-m3: {mb:>8.0f} ms  (διάμεσος)")
    print(f"  {args.model.split('/')[-1]:<18} {mn:>8.0f} ms  -> {mb/mn:.1f}x ταχύτερο")
    print(f"  εξοικονόμηση ανά ερώτηση: {(mb-mn)/1000:.1f}s")
    print(f"  με RERANK_CANDIDATES=50 το νέο θα έπαιρνε ~{mn*50/15:.0f} ms")

    print("\n" + "=" * 88)
    print("ΒΑΘΜΟΝΟΜΗΣΗ GATE — η κλίμακα ΑΛΛΑΖΕΙ, το 0.05 ΔΕΝ μεταφέρεται")
    print("=" * 88)
    for name, key in (("bge", "best_bge"), ("νέο", "best_new")):
        i_lo = min(r[key] for r in inc)
        o_hi = max(r[key] for r in ooc) if ooc else float("nan")
        print(f"  {name:<4} in-corpus  min={i_lo:>8.3f}  "
              f"διάμεσος={statistics.median(r[key] for r in inc):>8.3f}")
        print(f"  {name:<4} out-corpus max={o_hi:>8.3f}  "
              f"-> διάκενο {i_lo - o_hi:>+8.3f} "
              f"{'ΚΑΘΑΡΟ' if i_lo > o_hi else '*** ΕΠΙΚΑΛΥΨΗ — το gate θα κόβει σωστές ***'}")
        if ooc and i_lo > o_hi:
            print(f"  {name:<4} προτεινόμενο κατώφλι: {(i_lo + o_hi) / 2:.3f}")


asyncio.run(main())
