"""RRF με ΕΓΓΥΗΜΕΝΕΣ ΘΕΣΕΙΣ ανά σκέλος — δοκιμή δίπλα στην τρέχουσα, χωρίς αλλαγή κώδικα.

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΣΤΟΧΕΥΕΙ
-------------------------
Το `_rrf_fuse` δίνει σε ό,τι λείπει από μια λίστα rank **1000**:

    rrf_score = sum(w / (k + rm.get(doc_id, 1000) + 1) ...)

Άρα chunk που το βρήκε ΜΟΝΟ το dense στη θέση 15 παίρνει
    1/(60+16) + 1/(60+1001) = 0.01316 + 0.00094 = 0.01410
ενώ chunk στη θέση 15 ΚΑΙ στις δύο λίστες παίρνει
    2 * 1/(60+16)                                = 0.02632
Δηλαδή το δεύτερο σκέλος συνεισφέρει **28x** λιγότερο όταν λείπει. Αυτός είναι
ο ΑΚΡΙΒΗΣ μηχανισμός που κόβει το h002 (trace_query: dense θέση 15, BM25 απών,
RRF κομμένο) **πριν** το δει ο reranker.

Η ΠΑΡΑΛΛΑΓΗ: κρατάμε το RRF ως έχει, αλλά εγγυόμαστε ότι οι top-S κάθε σκέλους
μπαίνουν ΠΑΝΤΑ στους υποψηφίους. Μηδέν επιπλέον latency, μηδέν κλήσεις API,
καμία νέα εξάρτηση. Οι υπόλοιπες θέσεις γεμίζουν με τη σειρά του RRF.

ΤΙ ΜΕΤΡΑΕΙ, ΜΕ ΑΥΤΗ ΤΗ ΣΕΙΡΑ
-----------------------------
1. CONTROL: η δική μας υλοποίηση RRF πρέπει να δίνει ΤΑΥΤΟΣΗΜΟ top-N με το
   `ai_core._rrf_fuse`. Αν όχι, σταματάμε — δεν συγκρίνουμε τίποτα.
2. Υποψήφιοι: σε πόσες ερωτήσεις αλλάζει το σύνολο των 15.
3. ΣΕΛΙΔΕΣ: σε πόσες αλλάζει το τελικό prompt. Αυτό είναι το κριτήριο
   απόρριψης — το N=12 απορρίφθηκε στις 42/56.
4. Κάλυψη keywords στις σελίδες: φέρνει ΠΕΡΙΣΣΟΤΕΡΟ σωστό υλικό ή απλώς άλλο;
5. Gate: αλλάζει η απόφαση σιωπής σε καμία ερώτηση;

Η κάλυψη εδώ υπολογίζεται τοπικά (substring, πεζά) και ΔΕΝ είναι απαραίτητα
bit-exact με το eval_engine. Δεν πειράζει: συγκρίνονται A και B με την ΙΔΙΑ
συνάρτηση, άρα η διαφορά είναι έγκυρη.

    docker compose exec backend python evaluation/probe_rrf_slots.py
    docker compose exec backend python evaluation/probe_rrf_slots.py --slots 2
    docker compose exec backend python evaluation/probe_rrf_slots.py --csv evaluation/runs/rrf_slots.csv
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
        "golden_hard_paraphrase.jsonl"]


def load_sets(only=None):
    out = []
    for name in SETS:
        if only and name != only:
            continue
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = name
                    out.append(t)
    return out


def rrf_ids(dense_ids, sparse_ids, pos, k=60, top_n=15):
    """Ίδια αριθμητική με το _rrf_fuse, αλλά επιστρέφει ids (τα χρειαζόμαστε)."""
    lists = [dense_ids, sparse_ids]
    rank_maps = [{c: r for r, c in enumerate(lst)} for lst in lists]
    unique = list(dict.fromkeys([c for lst in lists for c in lst]))
    scored = [(sum(1.0 / (k + rm.get(c, 1000) + 1) for rm in rank_maps), c)
              for c in unique]
    scored.sort(key=lambda x: (-x[0], x[1]))   # ίδιο tie-break: -score, μετά id
    return [c for _s, c in scored[:top_n]]


def rrf_ids_slots(dense_ids, sparse_ids, pos, k=60, top_n=15, slots=3):
    """ΠΑΡΑΛΛΑΓΗ: οι top-`slots` κάθε σκέλους μπαίνουν πάντα, με τη σειρά τους.
    Οι υπόλοιπες θέσεις γεμίζουν από τη σειρά του κανονικού RRF."""
    forced = []
    for i in range(slots):                     # εναλλάξ dense/sparse ώστε να μην
        for lst in (dense_ids, sparse_ids):    # κυριαρχεί το ένα σκέλος στην αρχή
            if i < len(lst) and lst[i] not in forced:
                forced.append(lst[i])
    rest = [c for c in rrf_ids(dense_ids, sparse_ids, pos, k, top_n=10**9)
            if c not in forced]
    return (forced + rest)[:top_n]


def coverage(text: str, keywords) -> float:
    if not keywords:
        return float("nan")
    low = text.lower()
    return 100.0 * sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


def pageset(pages):
    return {(m["file_name"], m["page"]) for _t, m in pages}


async def run_variant(cand_ids, idx, query, keywords):
    """rerank -> gate -> σελίδες -> κάλυψη, για ένα σύνολο υποψηφίων."""
    pos, texts, metas = idx["pos"], idx["texts"], idx["metas"]
    items = [(texts[pos[c]], metas[pos[c]]) for c in cand_ids]
    pairs = [[query, t] for t, _m in items]
    scores = await asyncio.to_thread(
        lambda: ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE))
    ranked = sorted(zip([float(s) for s in scores], [t for t, _ in items],
                        [m for _, m in items]), key=lambda x: x[0], reverse=True)
    best = ranked[0][0]
    pages = await asyncio.to_thread(
        ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
    cov = coverage("\n".join(t for t, _m in pages), keywords)
    return best, pageset(pages), cov


async def main(args) -> int:
    tests = load_sets(args.only)
    if args.limit:
        tests = tests[:args.limit]
    print(f"{len(tests)} ερωτήσεις · slots={args.slots} · "
          f"RERANK_CANDIDATES={ai_core.RERANK_CANDIDATES} · gate={ai_core.MIN_RERANK_SCORE}\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    control_ok = True
    rows = []
    for i, t in enumerate(tests, 1):
        q = await ai_core.optimize_query(t["question"])
        dense_ids = ai_core._dense_exact_ids(
            dm, q, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
        sparse_ids = ai_core._bm25_sparse_ids(
            idx, q, allowed_ids, ai_core.DENSE_CANDIDATES)

        a_ids = rrf_ids(dense_ids, sparse_ids, idx["pos"],
                        top_n=ai_core.RERANK_CANDIDATES)
        b_ids = rrf_ids_slots(dense_ids, sparse_ids, idx["pos"],
                              top_n=ai_core.RERANK_CANDIDATES, slots=args.slots)

        # --- CONTROL: η δική μας RRF == η παραγωγική; --------------------
        prod = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                                 idx["metas"], k=60,
                                 top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
        prod_pages = [(m.get("file_name"), m.get("page")) for _s, _t, m in prod]
        ours_pages = [(idx["metas"][idx["pos"][c]].get("file_name"),
                       idx["metas"][idx["pos"][c]].get("page")) for c in a_ids]
        if prod_pages != ours_pages:
            control_ok = False
            print(f"  *** CONTROL ΑΠΕΤΥΧΕ στο {t['id']} — η υλοποίηση δεν ταιριάζει")

        kw = t.get("keywords") or []
        a_best, a_pages, a_cov = await run_variant(a_ids, idx, q, kw)
        b_best, b_pages, b_cov = await run_variant(b_ids, idx, q, kw)

        rows.append({
            "id": t["id"], "set": t["_set"], "category": t.get("category", ""),
            "cand_same": a_ids == b_ids,
            "pages_same": a_pages == b_pages,
            "a_best": a_best, "b_best": b_best,
            "a_cov": a_cov, "b_cov": b_cov,
            "a_cut": a_best < ai_core.MIN_RERANK_SCORE,
            "b_cut": b_best < ai_core.MIN_RERANK_SCORE,
            "n_only_a": len(a_pages - b_pages), "n_only_b": len(b_pages - a_pages),
        })
        if i % 15 == 0:
            print(f"  ...{i}/{len(tests)}", flush=True)

    if not control_ok:
        print("\n*** Το control απέτυχε — ΜΗΝ εμπιστευτείς τη σύγκριση.")
        return 1
    print("CONTROL OK: η υλοποίηση RRF του probe δίνει ταυτόσημο top-N με την παραγωγική.\n")

    # ------------------------------------------------------------ αναφορά
    print("=" * 78)
    n = len(rows)
    cand_diff = [r for r in rows if not r["cand_same"]]
    page_diff = [r for r in rows if not r["pages_same"]]
    gate_diff = [r for r in rows if r["a_cut"] != r["b_cut"]]
    print(f"υποψήφιοι άλλαξαν : {len(cand_diff)}/{n}")
    print(f"ΣΕΛΙΔΕΣ άλλαξαν   : {len(page_diff)}/{n}   <-- το κριτήριο απόρριψης")
    print(f"απόφαση gate άλλαξε: {len(gate_diff)}/{n}")

    have_kw = [r for r in rows if r["a_cov"] == r["a_cov"]]  # όχι NaN
    if have_kw:
        da = sum(r["a_cov"] for r in have_kw) / len(have_kw)
        db = sum(r["b_cov"] for r in have_kw) / len(have_kw)
        print(f"\nκάλυψη keywords στις σελίδες: τρέχον {da:.2f}%  ->  παραλλαγή {db:.2f}%"
              f"   (Δ {db - da:+.2f}pp)")
        better = [r for r in have_kw if r["b_cov"] > r["a_cov"]]
        worse = [r for r in have_kw if r["b_cov"] < r["a_cov"]]
        print(f"  καλύτερες {len(better)} · χειρότερες {len(worse)} · "
              f"ίδιες {len(have_kw) - len(better) - len(worse)}")
        for r in sorted(better + worse, key=lambda r: r["a_cov"] - r["b_cov"]):
            print(f"    {r['id']:<6} {r['set'][:18]:<18} {r['a_cov']:6.1f} -> {r['b_cov']:6.1f}"
                  f"   best {r['a_best']:+.2f} -> {r['b_best']:+.2f}")

    if gate_diff:
        print("\nΕΡΩΤΗΣΕΙΣ ΟΠΟΥ ΑΛΛΑΞΕ Η ΣΙΩΠΗ:")
        for r in gate_diff:
            print(f"    {r['id']:<6} {'κόβεται' if r['a_cut'] else 'περνάει'} -> "
                  f"{'κόβεται' if r['b_cut'] else 'περνάει'}"
                  f"   ({r['a_best']:+.2f} -> {r['b_best']:+.2f})")
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RRF με εγγυημένες θέσεις ανά σκέλος.")
    ap.add_argument("--slots", type=int, default=3,
                    help="πόσοι top κάθε σκέλους μπαίνουν πάντα (default 3)")
    ap.add_argument("--only", default=None, help="μόνο ένα σύνολο, π.χ. golden_set_50.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
