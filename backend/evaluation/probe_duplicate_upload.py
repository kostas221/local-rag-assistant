"""ΤΙ ΚΟΣΤΙΖΕΙ ΕΝΑ ΔΙΠΛΟ UPLOAD — προσομοίωση, ΜΗΔΕΝ άγγιγμα της βάσης.

ΤΟ ΠΡΟΒΛΗΜΑ
-----------
Το `/upload` δεν κάνει κανέναν έλεγχο διπλότυπου. Αν ανέβει το ίδιο PDF δύο
φορές, παίρνει **άλλο doc_id** και ξανα-ingestάρεται ολόκληρο. Το κρίσιμο δεν
είναι ο δίσκος — είναι ότι το `_expand_to_pages` κλειδώνει τις σελίδες σε

    key = (meta["doc_id"], meta["page"], meta["file_name"])

άρα η ΙΔΙΑ σελίδα με δύο doc_id είναι **δύο διαφορετικά κλειδιά**: μπαίνει δύο
φορές στο prompt. Και το `MAX_PAGES_PER_FILE` δεν βοηθά — η προεπιλογή του
είναι ΙΣΗ με το `MAX_PAGES` (8), δηλαδή κανένα πρακτικό όριο.

ΠΩΣ ΠΡΟΣΟΜΟΙΩΝΕΤΑΙ ΣΩΣΤΑ
-------------------------
Ένα ακριβές αντίγραφο δίνει, σε ΚΑΘΕ στάδιο, chunk με ταυτόσημο embedding,
ταυτόσημο BM25 σκορ και ταυτόσημο logit reranker. Άρα κάθε υποψήφιος
εμφανίζεται δύο φορές, ο ένας ακριβώς δίπλα στον άλλο:

    RERANK_CANDIDATES=15  ->  8 ΜΟΝΑΔΙΚΑ chunks (το 8ο μία φορά)
    EXPAND_INPUT=12       ->  6 ΜΟΝΑΔΙΚΑ chunks
    MAX_PAGES=8           ->  4 ΜΟΝΑΔΙΚΕΣ σελίδες, κάθε μία δύο φορές

Δεν χρειάζεται πλαστό ingest: τρέχουμε το ΠΡΑΓΜΑΤΙΚΟ pipeline μέχρι τον
reranker και μετά διπλασιάζουμε τη λίστα των υποψηφίων στη μνήμη.

Η επιλογή σελίδων αντιγράφεται τοπικά (`_select_keys`) επειδή το πραγματικό
`_expand_to_pages` κάνει `collection.get()` και δεν θα έβρισκε το πλαστό
doc_id. Γι' αυτό υπάρχει CONTROL: για το σενάριο Α τα κλειδιά μας πρέπει να
βγαίνουν ΤΑΥΤΟΣΗΜΑ με του παραγωγικού κώδικα. Αν όχι, σταματάμε.

ΠΡΟΒΛΕΨΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (καταγράφεται επίτηδες — δύο probes την έχουν ήδη
διαψεύσει): μοναδικές σελίδες 8 -> 4, κάλυψη πέφτει, χαρακτήρες prompt ~ίδιοι
ή λίγο κάτω (οι 4 σελίδες μετρημένες δύο φορές).

    docker compose exec backend python evaluation/probe_duplicate_upload.py
    docker compose exec backend python evaluation/probe_duplicate_upload.py \
        --csv evaluation/runs/duplicate_upload.csv
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

if hasattr(sys.stdout, "reconfigure"):     # Windows console -> cp1252
    sys.stdout.reconfigure(encoding="utf-8")

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_hard_paraphrase.jsonl"]


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


def _select_keys(top_chunks, max_pages):
    """ΑΚΡΙΒΕΣ αντίγραφο της επιλογής σελίδων του `_expand_to_pages`
    (σκορ σελίδας -> ταξινόμηση -> per-file cap -> max_pages), χωρίς το
    collection.get(). Ελέγχεται από το CONTROL παρακάτω."""
    by_page = {}
    for score, _text, meta in top_chunks:
        key = (meta.get("doc_id"), meta.get("page"), meta.get("file_name"))
        by_page.setdefault(key, []).append(float(score))

    ranked = sorted(
        by_page.items(),
        key=lambda kv: sum(sorted(kv[1], reverse=True)[:ai_core.PAGE_SCORE_TOP_K]),
        reverse=True)

    seen, per_file = [], {}
    for key, _scores in ranked:
        fname = key[2]
        if per_file.get(fname, 0) >= ai_core.MAX_PAGES_PER_FILE:
            continue
        seen.append(key)
        per_file[fname] = per_file.get(fname, 0) + 1
        if len(seen) >= max_pages:
            break
    return seen


def _duplicate(ranked):
    """Κάθε chunk δύο φορές, το αντίγραφο με άλλο doc_id — όπως ακριβώς θα
    ερχόταν από δεύτερο ingest του ίδιου PDF. Ίδιο σκορ -> γειτονικά."""
    out = []
    for score, text, meta in ranked:
        dup = dict(meta)
        dup["doc_id"] = (meta.get("doc_id") or 0) + 100000
        out.append((score, text, meta))
        out.append((score, text, dup))
    return out


def coverage(text, keywords):
    if not keywords:
        return float("nan")
    low = text.lower()
    return 100.0 * sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


async def page_texts(keys, ranked):
    """Το ΠΡΑΓΜΑΤΙΚΟ κείμενο των σελίδων: περνάμε στο _expand_to_pages μόνο τα
    chunks που ανήκουν σε αυτά τα κλειδιά (τα αντίγραφα έχουν εξ ορισμού το
    ίδιο κείμενο, οπότε αρκούν τα γνήσια)."""
    want = {(k[1], k[2]) for k in keys}          # (page, file_name)
    picked = [c for c in ranked
              if (c[2].get("page"), c[2].get("file_name")) in want]
    if not picked:
        return []
    return await asyncio.to_thread(
        ai_core._expand_to_pages, picked, len(want))


async def main(args) -> int:
    tests = load_sets(args.only)
    if args.limit:
        tests = tests[:args.limit]
    print(f"{len(tests)} ερωτήσεις · RERANK_CANDIDATES="
          f"{ai_core.RERANK_CANDIDATES} · EXPAND_INPUT={ai_core.EXPAND_INPUT} · "
          f"MAX_PAGES={ai_core.MAX_PAGES} · "
          f"MAX_PAGES_PER_FILE={ai_core.MAX_PAGES_PER_FILE}\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    control_ok, rows = True, []
    for i, t in enumerate(tests, 1):
        q = await ai_core.optimize_query(t["question"])
        dense_ids = ai_core._dense_exact_ids(
            dm, q, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
        sparse_ids = ai_core._bm25_sparse_ids(
            idx, q, allowed_ids, ai_core.DENSE_CANDIDATES)
        cands = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                                  idx["metas"], k=60,
                                  top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])

        pairs = [[q, c[1]] for c in cands]
        scores = await asyncio.to_thread(
            ai_core.reranker.predict, pairs,
            batch_size=ai_core.RERANK_BATCH_SIZE)
        ranked = sorted(zip([float(s) for s in scores],
                            [c[1] for c in cands], [c[2] for c in cands]),
                        key=lambda x: x[0], reverse=True)

        # --- Α: σήμερα -------------------------------------------------
        a_keys = _select_keys(ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

        # CONTROL: τα κλειδιά μας == του παραγωγικού κώδικα;
        prod = await asyncio.to_thread(
            ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT],
            ai_core.MAX_PAGES)
        if [(m["file_name"], m["page"]) for _t, m in prod] != \
           [(k[2], k[1]) for k in a_keys]:
            control_ok = False
            print(f"  *** CONTROL ΑΠΕΤΥΧΕ στο {t['id']}")

        # --- Β: το ίδιο PDF ανεβασμένο δύο φορές -----------------------
        dup = _duplicate(ranked)[:ai_core.RERANK_CANDIDATES]
        b_keys = _select_keys(dup[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
        b_uniq = {(k[1], k[2]) for k in b_keys}

        kw = t.get("keywords") or []
        a_pages = prod
        b_pages = await page_texts(b_keys, ranked)
        a_txt = "\n".join(x for x, _m in a_pages)
        b_txt = "\n".join(x for x, _m in b_pages)
        # Το πραγματικό prompt του Β περιέχει τις μοναδικές σελίδες ΟΣΕΣ
        # φορές εμφανίζονται ως κλειδί.
        mult = {}
        for k in b_keys:
            mult[(k[1], k[2])] = mult.get((k[1], k[2]), 0) + 1
        b_chars = sum(len(x) * mult.get((m["page"], m["file_name"]), 1)
                      for x, m in b_pages)

        rows.append({
            "id": t["id"], "set": t["_set"], "category": t.get("category", ""),
            "a_pages": len(a_pages), "b_slots": len(b_keys),
            "b_unique_pages": len(b_uniq),
            "lost_pages": len(a_pages) - len(b_uniq),
            "a_cov": coverage(a_txt, kw), "b_cov": coverage(b_txt, kw),
            "a_chars": len(a_txt), "b_chars": b_chars,
        })
        if i % 15 == 0:
            print(f"  ...{i}/{len(tests)}", flush=True)

    if not control_ok:
        print("\n*** Το control απέτυχε — ΜΗΝ εμπιστευτείς τη σύγκριση.")
        return 1

    n = len(rows)
    lost = sum(r["lost_pages"] for r in rows) / n
    cov_a = [r["a_cov"] for r in rows if r["a_cov"] == r["a_cov"]]
    cov_b = [r["b_cov"] for r in rows if r["b_cov"] == r["b_cov"]]
    worse = sum(1 for r in rows if r["b_cov"] < r["a_cov"])
    same = sum(1 for r in rows if r["b_cov"] == r["a_cov"])
    ch_a = sum(r["a_chars"] for r in rows) / n
    ch_b = sum(r["b_chars"] for r in rows) / n

    print("\n" + "=" * 66)
    print("ΤΟ ΙΔΙΟ PDF ΑΝΕΒΑΣΜΕΝΟ ΔΥΟ ΦΟΡΕΣ")
    print("=" * 66)
    print(f"CONTROL: η επιλογή σελίδων ταιριάζει με τον παραγωγικό κώδικα "
          f"σε {n}/{n}")
    print(f"μοναδικές σελίδες στο prompt   "
          f"{sum(r['a_pages'] for r in rows) / n:.2f} -> "
          f"{sum(r['b_unique_pages'] for r in rows) / n:.2f}   "
          f"(χάνονται {lost:.2f} ανά ερώτηση)")
    print(f"κάλυψη keywords                "
          f"{sum(cov_a) / len(cov_a):.2f}% -> {sum(cov_b) / len(cov_b):.2f}%   "
          f"(χειρότερα {worse}/{n} · ίδια {same}/{n})")
    print(f"χαρακτήρες prompt              {ch_a:.0f} -> {ch_b:.0f}   "
          f"({100 * (ch_b - ch_a) / ch_a:+.1f}%)")
    print("\nΤο μισό context είναι κυριολεκτικά το ΙΔΙΟ κείμενο δύο φορές, "
          "και οι\nθέσεις που καταλαμβάνει τις χάνουν πραγματικές σελίδες.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {args.csv}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="μόνο ένα σετ (όνομα αρχείου)")
    p.add_argument("--limit", type=int)
    p.add_argument("--csv")
    sys.exit(asyncio.run(main(p.parse_args())))
