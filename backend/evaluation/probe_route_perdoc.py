"""Χρειάζεται καν το Gemini για να σπάσει την ερώτηση; Αναζήτηση ΑΝΑ ΕΓΓΡΑΦΟ — Φάση 2, βήμα 6, 0 κλήσεις.

ΑΦΟΡΜΗ (probe_route_names.py, 30/9/2026): ο κανόνας «η ερώτηση ονομάζει ≥ 2 έγγραφα» πιάνει 28/29 multi_hop
με 0/170 ψευδώς θετικά, και κρατά ΟΛΟ το κέρδος της B (σωστά μισά 43/58). Η B όμως θέλει μια κλήση
Gemini για τα υπο-ερωτήματα. Αφού ο κανόνας ξέρει ΠΟΙΑ δύο έγγραφα ζητούνται, ίσως αρκεί να ψάξουμε
ΤΗΝ ΙΔΙΑ ερώτηση ΜΕΣΑ σε κάθε έγγραφο χωριστά: εκεί ανταγωνίζονται μόνο σελίδες του ίδιου paper, άρα
το ένα θέμα δεν μπορεί να πιάσει τις θέσεις του άλλου — ο κύριος μηχανισμός απώλειας του trace.
ΚΙΝΔΥΝΟΣ: μέσα στο έγγραφο κρίνει ΟΛΗ η ερώτηση. Το trace έδειξε ότι σε 7/9 «κανένα σκέλος» το σύστημα
βρίσκει το paper αλλά όχι τη σελίδα, και τα υπο-ερωτήματα της B είναι εστιασμένα ενώ εδώ δεν είναι.

ΤΙ ΚΑΝΕΙ (0 κλήσεις Gemini — μεταφράσεις ΠΑΓΩΜΕΝΕΣ, miss = σφάλμα, όπως το trace_multihop): για κάθε
ερώτηση που δρομολογεί ο κανόνας ονομάτων (probe_route_names), ίδιο 1ο πέρασμα με την παραγωγή (βάση)
και μετά, για ΚΑΘΕ ονομαζόμενο έγγραφο, dense + BM25 + RRF + κριτής με ΤΟ ΙΔΙΟ ερώτημα, ΠΕΡΙΟΡΙΣΜΕΝΑ
στα κομμάτια του εγγράφου (ίδιες συναρτήσεις με την παραγωγή, άλλη λίστα επιτρεπτών). Συγχώνευση ΟΠΩΣ
η B και η D του probe_decomp_mh40:
    PB  βάση + έως 4 νέες σελίδες από τα έγγραφα, εναλλάξ με τη σειρά αναφοράς     ≤ 12 σελίδες
    PD  2 εγγυημένες θέσεις ανά έγγραφο (έως το μισό budget) + η βάση             8 σελίδες
Ο φύλακας κρίνει ΜΟΝΟ το αρχικό ερώτημα (κομμένη σήμερα = κομμένη). ΚΑΝΕΝΑΣ φύλακας ανά έγγραφο: το
έγγραφο το ΟΝΟΜΑΣΕ ο χρήστης· καταγράφεται όμως πόσα θα κόβονταν. Ερωτήσεις που ο κανόνας ΔΕΝ δρομολογεί
μένουν ΑΚΡΙΒΩΣ βάση (από το baseline.json, χωρίς ανάκτηση).
Σύγκριση με τη B / D του Gemini (runs/decomp_mh40.csv) στα mh_new (σελίδες-τεκμήρια) και mh_old (κάλυψη).
ΕΛΕΓΧΟΣ: σελίδες βάσης = baseline.json σε ΟΛΕΣ τις δρομολογημένες.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026) — βάση 30/58 · B(Gemini) 40 · D(Gemini) 37:
    PB  σελίδες-τεκμήρια 35-40/58 · PD 33-38/58 · «και τα δύο» PB 11-15/29
    PB χάνει σελίδα-τεκμήριο που έχει η B(Gemini) σε 3-7 ερωτήσεις (όπου το υπο-ερώτημα εστιάζει)
    έγγραφα κάτω από τον φύλακα: 5-15 από τα ~70 (ΟΛΗ η ερώτηση βαθμολογεί κομμάτια ΕΝΟΣ θέματος)
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    PB ≥ 38/58 (το πολύ 2 κάτω από τη B) ΚΑΙ κάτω όριο CI της διαφοράς με τη βάση > 0
        -> η κλήση Gemini ΠΕΦΤΕΙ: επόμενο ο έλεγχος απαντήσεων της PB (κλήσεις ΜΟΝΟ όπου οι σελίδες
           διαφέρουν από της B, γιατί οι υπόλοιπες είναι ήδη στο πάγωμα) και μετά η αλλαγή στην παραγωγή.
    PB < 38 -> μένει η B με τον κανόνα ονομάτων (ήδη μετρημένη: 43/58, μία κλήση ΜΟΝΟ όταν ονομάζονται
        δύο έγγραφα)· επόμενο η αλλαγή στην παραγωγή πίσω από διακόπτη.

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, runs/route_perdoc.csv · 0 κλήσεις Gemini):
    έλεγχος: σελίδες βάσης = baseline σε 41/41 δρομολογημένες που πέρασαν τον φύλακα ✓ (+2 κομμένες = βάση)
    mh_new σελίδες-τεκμήρια /58: βάση 30 · D(Gemini) 37 · B(Gemini) 40 · PD 31 · PB 42
        Δ PB − βάση +12 [+7, +17] · Δ PB − B(Gemini) +2 [−3, +7] (ίδιες μέσα στον θόρυβο)
        «και τα δύο papers» 7 -> 18/29 (B(Gemini) 14) · σελίδες μ.ό. PB 11.6 · η PB ΔΕΝ χάνει ποτέ
        σελίδα της βάσης· χειρότερη από τη B(Gemini) σε 3 (m039 m049 m063), καλύτερη σε 5.
    ΤΟ ΚΕΡΔΟΣ ΘΕΛΕΙ ΤΙΣ ΕΠΙΠΛΕΟΝ ΣΕΛΙΔΕΣ: η PD (8 σελίδες, 2 εγγυημένες ανά έγγραφο) μένει στο 31. Οι 2
        πρώτες σελίδες κάθε εγγράφου είναι συνήθως ΗΔΗ στη βάση· η σελίδα-τεκμήριο του 2ου εγγράφου
        κάθεται βαθύτερα (ΟΛΗ η ερώτηση κρίνει μέσα στο έγγραφο) και τη φέρνουν μόνο οι +4 της PB.
    έγγραφα κάτω από τον φύλακα: 8/82 (διάμεσος καλύτερου λογιτ 1.40).
    ⚠️ ΑΝΤΙΘΕΤΗ ΕΝΔΕΙΞΗ: mh_old (9 δρομολογημένες) κάλυψη λέξεων βάση 18 · PB 18 · B(Gemini) 21 /26 — εκεί
        η αναζήτηση ανά έγγραφο δεν προσθέτει ΤΙΠΟΤΑ, ενώ τα υπο-ερωτήματα προσθέτουν 3. Μικρό δείγμα,
        χονδρή μετρική (λέξεις, όχι σελίδες-τεκμήρια) — αλλά δεν αγνοείται. hard_new (5): 8 -> 11/14.
    ΚΑΝΟΝΑΣ: PB 42 ≥ 38 ✓ · CI > 0 ✓ · ≥ B(Gemini) − 2 ✓ -> Η ΚΛΗΣΗ GEMINI ΠΕΦΤΕΙ, με την επιφύλαξη του
        mh_old. ΑΠΟΦΑΣΗ (30/9): τέλος στα probes — η PB μπαίνει στην παραγωγή πίσω από διακόπτη
        (doc_routing.py + ENABLE_PERDOC στο ai_core) και ο έλεγχος απαντήσεων γίνεται ΜΙΑ φορά πάνω
        στον πραγματικό κώδικα (scoreboard.py + scoreboard_answers.py με ENABLE_PERDOC=1).
    ΠΡΟΒΛΕΨΗ: PB 35-40 ✗ (42) · PD 33-38 ✗ (31) · «και τα δύο» PB 11-15 ✗ (18) · PB < B(Gemini) σε 3-7 ✓ (3)
        · κάτω από τον φύλακα 5-15 ✓ (8). Δύο στις πέντε — υποτίμησα την PB, υπερεκτίμησα την PD.

    docker compose run --rm --no-deps -v eval_near_ooc_store:/tmp/eval_near_ooc_chroma backend \
        python -u evaluation/probe_route_perdoc.py
"""
import asyncio
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import bootstrap_ci
import eval_near_ooc as E
import probe_decomp_mh40 as P
import probe_route_names as R
import scoreboard as S

import gemini_rest

SETS = {"mh_new": "golden_multihop_v2.jsonl", "mh_old": "golden_multihop_new.jsonl",
        "main": "golden_set_50.jsonl", "hard_new": "golden_hard_new.jsonl"}
BASE = os.path.join(S.OUT_DIR, "baseline.json")
OUT = os.path.join(S.RUNS, "route_perdoc.csv")
TARGET_EV, MAX_GAP = 38, 2                                       # κανόνας (docstring)


class DocPipe(P.Pipe):
    """Το 1ο πέρασμα του P.Pipe + το ΙΔΙΟ πέρασμα περιορισμένο στα κομμάτια ενός εγγράφου."""

    def __init__(self, ai):
        super().__init__(ai)
        allowed = set(self.allowed)
        self.by_doc = defaultdict(list)          # με τη σειρά του idx -> ίδιες ισοβαθμίες κάθε φορά
        for i, m in zip(self.idx["ids"], self.idx["metas"]):
            if i in allowed:
                self.by_doc[m["file_name"]].append(i)

    def candidates_in(self, q: str, ids: list) -> list:
        a = self.ai
        d = a._dense_exact_ids(self.dm, q, ids, min(a.DENSE_CANDIDATES, len(ids)))
        s = a._bm25_sparse_ids(self.idx, q, ids, a.DENSE_CANDIDATES)
        return a._rrf_fuse(d, s, self.idx["ids"], self.idx["texts"], self.idx["metas"],
                           k=60, top_n=a.RERANK_CANDIDATES, pos=self.idx["pos"])


def mention_order(text: str, docs: set, aliases: dict) -> list:
    t = text.lower()
    first = {d: min(t.find(a) for a in aliases[d] if a in t) for d in docs}
    return sorted(docs, key=first.get)


def ev_hits(t: dict, keys: list) -> int | None:
    return sum(ep in keys for ep in t["evidence_pages"]) if t.get("evidence_pages") else None


async def run() -> int:
    with open(BASE, encoding="utf-8") as f:
        base_rows = {(r["set"], r["id"]): r for r in json.load(f)["rows"]}
    with open(os.path.join(S.RUNS, "decomp_mh40.csv"), encoding="utf-8-sig") as f:
        llm = {(r["set"], r["id"]): r for r in csv.DictReader(f)}
    aliases = R.load_aliases()
    tests = {sk: E.load_jsonl(os.path.join(E.HERE, fn)) for sk, fn in SETS.items()}
    routed = {}
    for sk, ts in tests.items():
        for t in ts:
            b = base_rows[(sk, t["id"])]
            text = f"{b['question']} || {b.get('translation') or ''}"
            docs = R.docs_named(text, aliases)
            if len(docs) >= 2:
                routed[(sk, t["id"])] = mention_order(text, docs, aliases)
    print(f"δρομολογούνται {len(routed)}: " + " · ".join(
        f"{sk} {sum(k[0] == sk for k in routed)}/{len(ts)}" for sk, ts in tests.items()), flush=True)

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    ai_core.ENABLE_CORRECTIVE = True
    ai_core._save_query_emb_cache = lambda: None

    async def no_gemini(prompt, **kw):
        raise RuntimeError("νέα κλήση Gemini — δεν επιτρέπεται σε αυτό το script")

    gemini_rest.generate_once = no_gemini
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)
    frozen.install()
    from probe_decomposition import covered  # ΜΕΤΑ το open_store: ίδιο ai_core

    pipe = DocPipe(ai_core)
    mp = ai_core.MAX_PAGES
    rows, mismatch, below_gate, n_docs = [], [], 0, 0
    for sk, ts in tests.items():
        for t in ts:
            b = base_rows[(sk, t["id"])]
            base_keys = [k for k in b["pages"].split(";") if k]
            kws = t.get("keywords") or []
            row = {"set": sk, "id": t["id"], "category": t.get("category", ""), "lang": b["lang"],
                   "outcome": b["outcome"], "routed": int((sk, t["id"]) in routed),
                   "docs": ";".join(routed.get((sk, t["id"]), [])), "n_kw": len(kws),
                   "base_n": len(base_keys), "base_cov": b["cov"], "base_both": b["both"],
                   "base_ev": ev_hits(t, base_keys), "base_pages": b["pages"]}
            pb_keys = pd_keys = base_keys
            pb_cov = pd_cov = b["cov"]
            pb_both = pd_both = b["both"]
            leg_best = []
            if row["routed"]:
                base, _b1, _b2, _rw, outcome = await E.retrieve(ai_core, t["question"])
                if ";".join(P.key(m) for _x, m in base) != b["pages"]:
                    mismatch.append(f"{sk}:{t['id']}")
                if outcome == "passed_gate":
                    q = ai_core._translation_cache.get(t["question"], t["question"])
                    legs = []
                    for doc in routed[(sk, t["id"])]:
                        sf = pipe.rerank(q, pipe.candidates_in(q, pipe.by_doc[doc]))
                        leg_best.append(round(sf[0][0], 2))
                        below_gate += not pipe.passes(sf)
                        n_docs += 1
                        legs.append(pipe.pages(sf, mp))
                    pb = base + P.round_robin(legs, 4, skip={P.key(m) for _x, m in base})
                    reserved = P.round_robin([lp[:2] for lp in legs], mp // 2)
                    taken = {P.key(m) for _x, m in reserved}
                    pd = reserved + [pg for pg in base if P.key(pg[1]) not in taken][:mp - len(reserved)]
                    pb_keys = [P.key(m) for _x, m in pb]
                    pd_keys = [P.key(m) for _x, m in pd]
                    pb_cov, pd_cov = covered(pb, kws), covered(pd, kws)
                    pb_both, pd_both = S.both_docs(t, pb), S.both_docs(t, pd)
            row |= {"leg_best": ";".join(map(str, leg_best)),
                    "PB_n": len(pb_keys), "PB_cov": pb_cov, "PB_both": pb_both, "PB_ev": ev_hits(t, pb_keys),
                    "PD_n": len(pd_keys), "PD_cov": pd_cov, "PD_both": pd_both, "PD_ev": ev_hits(t, pd_keys),
                    "PB_pages": ";".join(pb_keys), "PD_pages": ";".join(pd_keys)}
            lr = llm.get((sk, t["id"]))
            row |= {"LB_ev": lr and lr["B_ev"], "LB_cov": lr and lr["B_cov"], "LD_ev": lr and lr["D_ev"],
                    "LB_pages": lr and lr["B_pages"]}
            rows.append(row)
            if row["routed"]:
                print(f"  {sk:<8} {t['id']:<5} {row['lang']} {'/'.join(d.split('.')[0][:10] for d in routed[(sk, t['id'])])}"
                      f"  best {row['leg_best'] or '-':<12} | τεκμ. βάση {row['base_ev'] if row['base_ev'] is not None else '-'}"
                      f" PB {row['PB_ev'] if row['PB_ev'] is not None else '-'} B(Gemini) {row['LB_ev'] or '-'}"
                      f" | κάλυψη {row['base_cov']}->{row['PB_cov']}/{len(kws)}", flush=True)

    if frozen.errors or frozen.misses:
        print(f"!! {len(frozen.errors) + frozen.misses} κλήσεις έξω από το πάγωμα — ΔΕΝ ΜΕΤΡΑΕΙ")
        return 1
    report(rows, mismatch, below_gate, n_docs, frozen.hits)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Σώθηκε: {OUT}")
    return 1 if mismatch else 0


def report(rows, mismatch, below_gate, n_docs, hits) -> None:
    n_routed = sum(r["routed"] for r in rows)
    print("\n" + "#" * 100)
    print(f"ΕΛΕΓΧΟΣ: σελίδες βάσης = baseline.json σε {n_routed - len(mismatch)}/{n_routed} δρομολογημένες"
          + (f"  !! διαφέρουν: {mismatch}" if mismatch else "  ✓") + f" · Gemini 0 νέες ({hits} από το πάγωμα)")
    print(f"έγγραφα κάτω από τον φύλακα (ΟΛΗ η ερώτηση μέσα στο έγγραφο): {below_gate}/{n_docs}")
    new = [r for r in rows if r["set"] == "mh_new"]
    n = len(new)

    def tot(k):
        return sum(int(r[k]) for r in new if r[k] not in (None, ""))

    print(f"\nmh_new ({n}): σελίδες-τεκμήρια /{2 * n}: βάση {tot('base_ev')} · D(Gemini) {tot('LD_ev')} · "
          f"B(Gemini) {tot('LB_ev')} · PD {tot('PD_ev')} · PB {tot('PB_ev')}")
    print(f"  «και τα δύο papers» /{n}: βάση {tot('base_both')} · PD {tot('PD_both')} · PB {tot('PB_both')}"
          f"   ·   σελίδες μ.ό. PB {sum(r['PB_n'] for r in new) / n:.1f}")
    ci = {}
    for s in ("PB", "PD"):
        m, lo, hi = bootstrap_ci.paired_ci([(r["base_ev"], r[f"{s}_ev"]) for r in new], 10000, 42)
        ci[s] = lo
        print(f"  Δ {s} − βάση: {m * n:+.1f} [{lo * n:+.1f}, {hi * n:+.1f}]")
    m, lo, hi = bootstrap_ci.paired_ci([(int(r["LB_ev"]), r["PB_ev"]) for r in new], 10000, 42)
    print(f"  Δ PB − B(Gemini): {m * n:+.1f} [{lo * n:+.1f}, {hi * n:+.1f}]")
    lost = [r["id"] for r in new if r["PB_ev"] < int(r["LB_ev"])]
    won = [r["id"] for r in new if r["PB_ev"] > int(r["LB_ev"])]
    print(f"  PB χειρότερη από B(Gemini): {len(lost)} {lost} · καλύτερη: {len(won)} {won}")
    for sk in ("mh_old", "main", "hard_new"):
        rs = [r for r in rows if r["set"] == sk and r["routed"]]
        if rs:
            lb = sum(int(r["LB_cov"]) for r in rs if r["LB_cov"] not in (None, ""))
            print(f"{sk} (δρομολογημένες {len(rs)}): κάλυψη βάση {sum(r['base_cov'] for r in rs)} · PD "
                  f"{sum(r['PD_cov'] for r in rs)} · PB {sum(r['PB_cov'] for r in rs)} /{sum(r['n_kw'] for r in rs)}"
                  + (f" · B(Gemini) {lb}" if sk == "mh_old" else ""))
    pb, lb_ev = tot("PB_ev"), tot("LB_ev")
    ok = (pb >= TARGET_EV, ci["PB"] > 0, pb >= lb_ev - MAX_GAP)
    print(f"\nΚΑΝΟΝΑΣ: PB ≥ {TARGET_EV} {'✓' if ok[0] else '✗'} · CI > 0 {'✓' if ok[1] else '✗'} · "
          f"≥ B(Gemini) − {MAX_GAP} {'✓' if ok[2] else '✗'}  -> "
          + ("Η ΚΛΗΣΗ GEMINI ΠΕΦΤΕΙ (αναζήτηση ανά έγγραφο)" if all(ok) else
             "ΜΕΝΕΙ η B με τον κανόνα ονομάτων"))
    print("#" * 100)


def main() -> int:
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run())
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    sys.exit(main())
