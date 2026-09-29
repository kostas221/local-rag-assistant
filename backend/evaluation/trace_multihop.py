"""ΣΕ ΠΟΙΟ ΣΤΑΔΙΟ χάνεται η σελίδα-τεκμήριο στις multi_hop; — Φάση 2, βήμα 1, ΜΗΔΕΝ κλήσεις Gemini.

ΑΦΟΡΜΗ (scoreboard baseline, 28/9/2026): στις 29 νέες multi_hop (golden_multihop_v2) η σελίδα με
το ΣΤΟΙΧΕΙΟ έρχεται 30/58· του paper που αναφέρεται 1ο στην ερώτηση 19/29, του 2ου 11/29.
Το probe_multihop_lang έδειξε ότι ΔΕΝ φταίει κυρίως η μετάφραση. Δεν ξέρουμε ΠΟΥ χάνεται — και
κάθε διόρθωση διορθώνει άλλο στάδιο:
    κανένα σκέλος δεν τη βρίσκει (dense 30 / BM25 30)   -> διάσπαση ερώτησης ή γράφος (Φάση 2)
    κόβεται στη συγχώνευση (RRF θέση > 15)               -> RERANK_CANDIDATES↑ (μόνο ρύθμιση)
    ο κριτής τη βάζει κάτω από τα 12 (EXPAND_INPUT)      -> μοίρασμα θέσεων ανά paper
    κόβεται στο όριο 8 σελίδων                            -> μοίρασμα θέσεων ανά paper

ΤΙ ΚΑΝΕΙ: ο ΠΡΑΓΜΑΤΙΚΟΣ search_documents, απομονωμένο store (418), Gemini παγωμένο όπως στο
scoreboard (runs/scoreboard_gemini.json) — ΑΠΑΓΟΡΕΥΜΕΝΗ κάθε νέα κλήση: αν λείπει κάτι από το
πάγωμα το script σταματάει, δεν ψάχνει αμετάφραστα. Οι εσωτερικές συναρτήσεις τυλίγονται και
καταγράφουν τι βγάζουν (ΚΑΜΙΑ αλλαγή στη λογική): dense, BM25, RRF (ολόκληρη η λίστα, όχι μόνο
τα 15), κριτής, 12 καλύτερα, 8 σελίδες. Αν πέρασε μέσω corrective, μετράει το 2ο πέρασμα (από
εκεί βγήκαν οι σελίδες). Σειρά αναφοράς paper: τα ίδια ψευδώνυμα με την ανάλυση της 28/9, πάνω
στο question_en.
Έλεγχος ακεραιότητας: οι σελίδες που βγαίνουν εδώ πρέπει να είναι ΤΑΥΤΟΣΗΜΕΣ με του
baseline.json σε 29/29 — αλλιώς δεν μετράμε το ίδιο σύστημα.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (29/9/2026):
    χαμένες σελίδες-τεκμήρια 28/58 (όσες λέει η βάση)
    ΠΡΙΝ τον κριτή (κανένα σκέλος + συγχώνευση)  ≥ 18/28
        από αυτές «κανένα σκέλος» ≥ 8
    κριτής + όριο σελίδων                         ≤ 10/28
    του 2ου paper χάνονται περισσότερες «σε κανένα σκέλος» από του 1ου
    Σκεπτικό: ο κριτής πετάει μόνο 3 θέσεις (15 -> 12) — δομικά δεν μπορεί να είναι ο κύριος
    ένοχος· οι σελίδες του 1ου θέματος πιάνουν ήδη τις θέσεις στο dense ΚΑΙ στο BM25.
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    κυρίως «συγχώνευση» με θέση RRF ≤ 20  -> μετράμε RERANK_CANDIDATES=20 (+~260 ms, μέσα στο 1.2 s)
    κυρίως «κανένα σκέλος»                -> περισσότερα υποψήφια ΔΕΝ βοηθούν· διάσπαση ερώτησης
                                             (στρατηγική C, ξανά στο νέο σετ) ή HippoRAG 2
    κυρίως κριτής / όριο σελίδων          -> θέσεις ανά paper στις 8 σελίδες

ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/trace_multihop.csv): έλεγχος 29/29 ίδιες σελίδες με τη βάση ✓ · 0 Gemini.
    στάδιο απώλειας   όλες  1ο paper  2ο paper   el  en
    έφτασε              30        19        11   19  11
    κανένα σκέλος        9         5         4    9   0
    συγχώνευση          12         3         9   10   2
    κριτής               5         0         5    4   1
    όριο σελίδων         2         2         0    2   0
    ΠΡΙΝ τον κριτή 21/28 ✓ · «κανένα σκέλος» 9 ✓ · κριτής + όριο 7 ✓ · «2ο paper περισσότερα σε
    κανένα σκέλος» ✗ (5 έναντι 4). Τρεις στις τέσσερις.
    - «συγχώνευση»: θέσεις RRF 18-39, διάμεσος ~28· ≤20 ΜΟΝΟ 3, ≤25 5. Και 10/12 βρίσκονται σε ΕΝΑ
      μόνο σκέλος (π.χ. m049: BM25 θέση 3, λείπει από dense -> RRF 20) — ο μηχανισμός του h002.
    - «κανένα σκέλος»: 7/9 έχουν ΑΛΛΗ σελίδα του ίδιου paper στις 8 -> το σύστημα βρίσκει το
      paper, όχι τη σελίδα με το στοιχείο. Και οι 9 είναι ελληνικές ερωτήσεις.
    - το 2ο paper χάνεται στη ΜΑΧΗ ΘΕΣΕΩΝ (συγχώνευση 9 + κριτής 5), το 1ο όταν δεν βρεθεί καθόλου.
    - «κριτής»: θέσεις 14-15 από 15 — το στοιχείο του 2ου μισού βαθμολογείται με ΟΛΗ την ερώτηση.
    ΚΑΝΟΝΑΣ: ο κλάδος «RERANK_CANDIDATES=20» ΔΕΝ ενεργοποιείται — φτάνει το πολύ 3/28 (και αυτές
    πρέπει ακόμα να περάσουν τον κριτή). 21/28 χάνονται ΠΡΙΝ τον κριτή, με ΕΝΑ ερώτημα για δύο
    θέματα -> ο δρόμος είναι αναζήτηση ΑΝΑ ΜΕΡΟΣ της ερώτησης (διάσπαση, ξανά στο νέο σετ).
    ΦΤΑΝΕΙ ΣΤΗΝ ΑΠΑΝΤΗΣΗ (answers_baseline_mh_new.json, 28/9): σελίδα ήρθε -> σωστό 27/30,
    χωρίς στήριξη 0· σελίδα ΔΕΝ ήρθε -> σωστό 6/28, χωρίς στήριξη 4. Η ανάκτηση είναι ο μοχλός.

    docker compose exec backend python evaluation/trace_multihop.py
"""
import asyncio
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as SB

import gemini_rest

MH = os.path.join(E.HERE, "golden_multihop_v2.jsonl")
BASE = os.path.join(SB.OUT_DIR, "baseline.json")
OUT = os.path.join(SB.RUNS, "trace_multihop.csv")

# ίδια με την ανάλυση σειράς αναφοράς της 28/9 (19/29 έναντι 11/29)
ALIAS = {
    "1702.04024.pdf": r"pywren|occupy the cloud|jonas et al\. ?\(?2017",
    "1706.03178.pdf": r"baldini",
    "1812.03651.pdf": r"cidr|hellerstein|one step forward",
    "1902.03383v1.pdf": r"berkeley view on serverless|berkeley \(2019\)|berkeley view \(2019\)|"
                        r"2019 berkeley|cloud programming simplified|jonas et al|berkeley's 2019|"
                        r"berkeley 2019",
    "EECS-2009-28.pdf": r"above the clouds|berkeley view of cloud computing|2009",
    "excamera-nsdi17.pdf": r"excamera|\bmu\b",
    "mapreduce-osdi04.pdf": r"mapreduce",
}
STAGES = ["έφτασε", "κανένα σκέλος", "συγχώνευση", "κριτής", "όριο σελίδων", "σιωπή"]

_passes: list[dict] = []


def page_of(meta: dict) -> str:
    return f"{meta.get('file_name')}:{meta.get('page')}"


def install_tracers(ai_core) -> None:
    """Τυλίγει τα στάδια ΧΩΡΙΣ να αλλάζει τι επιστρέφουν. Κάθε κλήση dense ανοίγει νέο πέρασμα."""
    idx = ai_core._get_bm25_index()
    pos, metas = idx["pos"], idx["metas"]
    o_dense, o_bm25 = ai_core._dense_exact_ids, ai_core._bm25_sparse_ids
    o_rrf, o_expand = ai_core._rrf_fuse, ai_core._expand_to_pages
    o_pred = ai_core.reranker.predict

    def pages_of_ids(ids):
        return [page_of(metas[pos[i]]) for i in ids]

    def dense(dm, query, allowed, top_n=30):
        out = o_dense(dm, query, allowed, top_n)
        _passes.append({"query": query, "dense": pages_of_ids(out)})
        return out

    def bm25(ix, query, allowed, top_n=30):
        out = o_bm25(ix, query, allowed, top_n)
        _passes[-1]["bm25"] = pages_of_ids(out)
        return out

    def rrf(*a, top_n=15, **kw):
        full = o_rrf(*a, top_n=10**6, **kw)           # ολόκληρη η συγχωνευμένη λίστα (30+30)
        _passes[-1]["rrf"] = [page_of(m) for _s, _t, m in full]
        return full[:top_n]                             # ό,τι θα έδινε με top_n (ίδια ταξινόμηση)

    def pred(pairs, **kw):
        scores = o_pred(pairs, **kw)
        # ίδια σειρά με την παραγωγή: sorted(..., reverse=True) είναι σταθερό στις ισοπαλίες
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand = _passes[-1]["rrf"][:len(pairs)]
        _passes[-1]["reranked"] = [(cand[i], float(scores[i])) for i in order]
        return scores

    def expand(top_chunks, max_pages=8, user_id=None):
        out = o_expand(top_chunks, max_pages, user_id)
        _passes[-1]["top12"] = [page_of(m) for _s, _t, m in top_chunks]
        _passes[-1]["final"] = [page_of(m) for _t, m in out]
        return out

    ai_core._dense_exact_ids, ai_core._bm25_sparse_ids = dense, bm25
    ai_core._rrf_fuse, ai_core._expand_to_pages = rrf, expand
    ai_core.reranker.predict = pred


def rank_in(pages: list[str], ep: str) -> int | None:
    return next((i + 1 for i, p in enumerate(pages) if p == ep), None)


def stage_of(p: dict | None, ep: str) -> str:
    if p is None or "final" not in p:
        return "σιωπή"
    if ep in p["final"]:
        return "έφτασε"
    if ep in p["top12"]:
        return "όριο σελίδων"
    if ep in [x for x, _s in p["reranked"]]:
        return "κριτής"
    if ep in p["rrf"]:
        return "συγχώνευση"
    return "κανένα σκέλος"


def mention_order(t: dict) -> dict:
    q = t["question_en"].lower()
    pos = {}
    for ep in t["evidence_pages"]:
        m = re.search(ALIAS[ep.rsplit(":", 1)[0]], q)
        pos[ep] = m.start() if m else None
    if any(v is None for v in pos.values()):
        return dict.fromkeys(t["evidence_pages"], "?")
    ordered = sorted(t["evidence_pages"], key=lambda e: pos[e])
    return {ep: ("1ο", "2ο")[i] for i, ep in enumerate(ordered)}


async def run() -> int:
    with open(BASE, encoding="utf-8") as f:
        base = {r["id"]: r for r in json.load(f)["rows"] if r["set"] == "mh_new"}
    tests = E.load_jsonl(MH)

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != SB.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {SB.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()          # μεταφράσεις ΜΟΝΟ από το πάγωμα, όπως το scoreboard
    ai_core.ENABLE_CORRECTIVE = True

    async def no_gemini(prompt, **kw):
        raise RuntimeError("νέα κλήση Gemini — δεν επιτρέπεται σε αυτό το script")

    gemini_rest.generate_once = no_gemini        # κάτω από το πάγωμα: miss -> σφάλμα, όχι κλήση
    frozen = SB.Frozen(SB.FROZEN_PATH, fresh=False)
    frozen.install()
    install_tracers(ai_core)

    rows, mismatch = [], []
    print(f"\n{len(tests)} multi_hop · στάδιο απώλειας κάθε σελίδας-τεκμηρίου\n")
    for t in tests:
        _passes.clear()
        pages, _b1, _b2, _rw, outcome = await E.retrieve(ai_core, t["question"])
        got = ";".join(f"{m.get('file_name')}:{m.get('page')}" for _x, m in pages)
        if got != base[t["id"]]["pages"]:
            mismatch.append(t["id"])
        p = _passes[-1] if _passes and "final" in _passes[-1] else None
        order = mention_order(t)
        final_docs = Counter(x.rsplit(":", 1)[0] for x in (p["final"] if p else []))
        line = []
        for ep in t["evidence_pages"]:
            doc = ep.rsplit(":", 1)[0]
            st = stage_of(p, ep)
            rr = [x for x, _s in p["reranked"]] if p else []
            sc = max((s for x, s in p["reranked"] if x == ep), default=None) if p else None
            cut12 = p["reranked"][len(p["top12"]) - 1][1] if p and p["top12"] else None
            rows.append({
                "id": t["id"], "lang": t["lang"], "outcome": outcome, "evidence": ep,
                "mention": order[ep], "stage": st,
                "dense_rank": rank_in(p["dense"], ep) if p else None,
                "bm25_rank": rank_in(p["bm25"], ep) if p else None,
                "rrf_rank": rank_in(p["rrf"], ep) if p else None,
                "rerank_rank": rank_in(rr, ep), "rerank_score": None if sc is None else round(sc, 2),
                "score_12th": None if cut12 is None else round(cut12, 2),
                "pages_same_doc": final_docs.get(doc, 0),
                "pages_other_evidence_doc": sum(final_docs.get(e.rsplit(":", 1)[0], 0)
                                                for e in t["evidence_pages"] if e != ep),
                "query": p["query"] if p else "", "final_pages": ";".join(p["final"]) if p else "",
            })
            r = rows[-1]
            line.append(f"{order[ep]} {st:<14}(d {r['dense_rank'] or '-':>2} b {r['bm25_rank'] or '-':>2}"
                        f" rrf {r['rrf_rank'] or '-':>2} κρ {r['rerank_rank'] or '-':>2})")
        print(f"  {t['id']} {t['lang']}  " + "  |  ".join(line), flush=True)

    if frozen.errors or frozen.misses:
        print(f"!! {len(frozen.errors)} κλήσεις έξω από το πάγωμα — ΤΟ ΤΡΕΞΙΜΟ ΔΕΝ ΜΕΤΡΑΕΙ ({frozen.errors[:1]})")
        return 1

    n = len(rows)
    lost = [r for r in rows if r["stage"] != "έφτασε"]
    print("\n" + "#" * 92)
    print(f"ΕΛΕΓΧΟΣ: σελίδες ταυτόσημες με το baseline σε {len(tests) - len(mismatch)}/{len(tests)}"
          + (f"  !! διαφέρουν: {mismatch}" if mismatch else "  ✓"))
    print(f"Gemini: 0 νέες κλήσεις · {frozen.hits} από το πάγωμα")
    print(f"\nσελίδες-τεκμήρια που έφτασαν: {n - len(lost)}/{n} · χάθηκαν {len(lost)}")
    print(f"\n{'στάδιο απώλειας':<18}{'όλες':>6}{'1ο paper':>10}{'2ο paper':>10}{'el':>6}{'en':>6}")
    for st in STAGES:
        rs = [r for r in rows if r["stage"] == st]
        print(f"{st:<18}{len(rs):>6}{sum(r['mention'] == '1ο' for r in rs):>10}"
              f"{sum(r['mention'] == '2ο' for r in rs):>10}"
              f"{sum(r['lang'] == 'el' for r in rs):>6}{sum(r['lang'] == 'en' for r in rs):>6}")
    pre = sum(r["stage"] in ("κανένα σκέλος", "συγχώνευση") for r in lost)
    post = sum(r["stage"] in ("κριτής", "όριο σελίδων") for r in lost)
    print(f"\nΠΡΙΝ τον κριτή {pre}/{len(lost)}  (πρόβλεψη ≥ 18) · "
          f"«κανένα σκέλος» {sum(r['stage'] == 'κανένα σκέλος' for r in lost)} (πρόβλεψη ≥ 8) · "
          f"κριτής + όριο {post} (πρόβλεψη ≤ 10)")
    fus = sorted(r["rrf_rank"] for r in lost if r["stage"] == "συγχώνευση")
    if fus:
        print(f"«συγχώνευση»: θέσεις RRF {fus} · ≤20: {sum(x <= 20 for x in fus)} · "
              f"≤25: {sum(x <= 25 for x in fus)}")
    crit = [r for r in lost if r["stage"] == "κριτής"]
    if crit:
        print("«κριτής»: " + " · ".join(f"{r['id']} θέση {r['rerank_rank']} ({r['rerank_score']} "
                                          f"έναντι 12ης {r['score_12th']})" for r in crit))
    none_ = [r for r in lost if r["stage"] == "κανένα σκέλος"]
    if none_:
        crowd = Counter("ίδιο paper" if r["pages_same_doc"] else "ΚΑΜΙΑ σελίδα του paper" for r in none_)
        print(f"«κανένα σκέλος»: στις 8 σελίδες υπάρχει άλλη σελίδα του ίδιου paper -> {dict(crowd)}")
    print("#" * 92)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Σώθηκε: {OUT}")
    return 1 if mismatch else 0


def main() -> int:
    try:
        fd = os.open(SB.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {SB.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run())
    finally:
        os.close(fd)
        os.remove(SB.LOCK)


if __name__ == "__main__":
    sys.exit(main())
