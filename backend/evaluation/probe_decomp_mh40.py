"""Φέρνει η ΔΙΑΣΠΑΣΗ της ερώτησης τη σελίδα-τεκμήριο στις multi_hop; — Φάση 2, βήμα 2.

ΑΦΟΡΜΗ (trace_multihop.py, 29/9/2026): στις 29 νέες multi_hop φτάνει στο Gemini η σελίδα με το
στοιχείο 30/58. Από τις 28 που χάνονται, οι 21 χάνονται ΠΡΙΝ τον κριτή (κανένα σκέλος 9 ·
συγχώνευση 12): ΕΝΑ ερώτημα για ΔΥΟ θέματα, το ένα πιάνει τις θέσεις. Και φτάνει στην απάντηση:
σελίδα ήρθε -> σωστό 27/30 · δεν ήρθε -> 6/28.

ΓΙΑΤΙ ΞΑΝΑ (απορρίφθηκε 8/2026 — probe_decomposition + probe_decomp_merge). Δύο λόγοι:
    (α) κανένα κέρδος, σε 11 multi_hop με απαντήσεις ΗΔΗ 5/5 -> δεν μπορούσε να φανεί. ΔΕΝ ισχύει εδώ.
    (β) κόστος: +1 κλήση Gemini σε ΚΑΘΕ ερώτηση (routing) + χρόνος. ΙΣΧΥΕΙ ακόμα. Αυτό το probe ΔΕΝ
        τον απαντάει — μετράει ΜΟΝΟ αν το κέρδος αξίζει να τον ξανασκεφτούμε (π.χ. η διάσπαση μέσα
        στην κλήση μετάφρασης, που υπάρχει ήδη στις ελληνικές).

ΤΙ ΚΑΝΕΙ: απομονωμένο store (418), Gemini της παραγωγής ΠΑΓΩΜΕΝΟ (scoreboard_gemini.json — καμία
νέα μετάφραση ή αναδιατύπωση· miss -> σφάλμα). ΝΕΕΣ κλήσεις ΜΟΝΟ για τη διάσπαση: 1 ανά ερώτηση, ΙΔΙΟ
prompt με τον Αύγουστο (probe_decomposition.DEFAULT_PROMPT: διάσπαση + routing σε μία κλήση), πάνω στο
ερώτημα που ψάχνει σήμερα η παραγωγή (τη μετάφραση, στις ελληνικές). Παγώνουν στο
runs/decomp_mh40_gemini.json -> 2ο τρέξιμο = 0 κλήσεις. ΚΑΜΙΑ αλλαγή στο ai_core. Το cache διανυσμάτων
ερωτήσεων της παραγωγής ΔΕΝ γράφεται (έχει όριο 1.000 — τα υπο-ερωτήματα θα έτρωγαν θέσεις χρηστών).

ΣΤΑΘΕΡΟ ΣΕ ΟΛΕΣ ΤΙΣ ΣΤΡΑΤΗΓΙΚΕΣ: ο ΦΥΛΑΚΑΣ κρίνει ΜΟΝΟ το αρχικό ερώτημα, όπως σήμερα. Ό,τι κόβεται
σήμερα ή περνάει μέσω corrective μένει ΑΚΡΙΒΩΣ όπως είναι -> η διάσπαση δεν μπορεί να ανοίξει
διαρροή (μάθημα της αποσαφήνισης 13/8: τα υπο-ερωτήματα κληρονομούν λεξιλόγιο του corpus).

ΣΤΡΑΤΗΓΙΚΕΣ (όλες ΜΗΔΕΝ κόστος μετά τη διάσπαση):
    βάση  ο πραγματικός search_documents                                           8 σελίδες
    C     ένωση υποψηφίων (αρχικό + σκέλη), ΕΝΑ rerank με το ΑΡΧΙΚΟ ερώτημα       8  ΚΥΡΙΑ — ορίστηκε
          ΠΡΙΝ, στον κανόνα του trace_multihop («στρατηγική C, ξανά στο νέο σετ»)
    D     εγγυημένες θέσεις: οι 2 καλύτερες σελίδες κάθε σκέλους (rerank με ΤΟ ΔΙΚΟ ΤΟΥ
          ερώτημα, έως το μισό budget), οι υπόλοιπες από τη βάση με τη σειρά της  8  ΝΕΑ
    A     ανά σκέλος εναλλάξ — η A του Αυγούστου, ΑΛΛΑ σκέλος κάτω από τον φύλακα δίνει τις
          θέσεις του στα άλλα (εκεί έχασε το 8/2026: q046 3/3 -> 0/3)            8  αναφορά
    B     βάση + έως 4 νέες από τα σκέλη (υπερσύνολο)                          ≤12  ταβάνι, +tokens
    Γιατί D: το trace δείχνει ότι η C ΔΕΝ μπορεί να σώσει τον «κριτή» (θέσεις 14-15 με ΟΛΗ την
    ερώτηση) — ο κριτής με το αρχικό ερώτημα ξαναθάβει ό,τι φέρει το σκέλος (ο μηχανισμός του h002).

ΜΕΤΡΙΚΕΣ: σελίδες-τεκμήρια (29 νέες, /58) · «και τα δύο papers» (/29) · κάλυψη λέξεων (40, /113) ·
ανά ερώτηση ΚΕΡΔΙΣΕ/ΕΧΑΣΕ σελίδα-τεκμήριο · ποιο ΣΤΑΔΙΟ απώλειας του trace σώζει κάθε στρατηγική ·
ζευγαρωτό bootstrap (10.000, seed 42) · χρόνος: επιπλέον ανάκτηση (CPU) + κλήση διάσπασης.
ΕΛΕΓΧΟΙ ΑΚΕΡΑΙΟΤΗΤΑΣ: (1) η βάση δίνει σελίδες ΤΑΥΤΟΣΗΜΕΣ με το baseline.json σε 40/40· (2) η δική
μου αναπαραγωγή του 1ου περάσματος δίνει τις ίδιες σελίδες με τη βάση (αλλιώς οι στρατηγικές
χτίζονται πάνω σε άλλο σύστημα).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (29/9/2026):
    βάση: 30/58 · δύο 7/29 · κάλυψη 66/113 (ό,τι λέει το baseline)
    routing: σπάει ≥ 36/40 (όλες ρωτάνε ρητά για δύο papers)
    C  34-38/58 — σώζει μέρος από «κανένα σκέλος»/«συγχώνευση», ΟΧΙ τον «κριτή»
    D  40-46/58 · χάνει σελίδα-τεκμήριο της βάσης σε ≤ 2 ερωτήσεις
    A  38-46/58 · αλλά κάλυψη των 11 παλιών κάτω από 23/32 (πετάει σελίδες της βάσης)
    B  ≥ D, με ~11 σελίδες
    επιπλέον χρόνος ανάκτησης (διάμεσος): σκέλη 0.3-0.8 s · C 0.4-1.0 s · κλήση διάσπασης 0.6-1.5 s
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    ΑΞΙΖΕΙ ΤΟ ΕΠΟΜΕΝΟ ΒΗΜΑ αν στρατηγική με ≤ 8 σελίδες (C, D ή A) πετυχαίνει ΟΛΑ:
        σελίδες-τεκμήρια ≥ 38/58 (+8) · κάτω όριο του ζευγαρωτού CI της διαφοράς > 0 ·
        κάλυψη των 11 παλιών ≥ 23/32 · χάνει σελίδα-τεκμήριο της βάσης σε ≤ 2 ερωτήσεις.
        Αν δεν είναι η C, διαλέχτηκε ανάμεσα σε τρεις στο ΙΔΙΟ δείγμα -> ΕΝΔΕΙΞΗ, επιβεβαιώνεται στον
        κριτή απαντήσεων. Επόμενο βήμα: routing σε ΟΛΑ τα σετ (ψευδώς θετικά), χρόνος, απαντήσεις,
        και αν χωράει η διάσπαση μέσα στην κλήση μετάφρασης.
    Αν το πετυχαίνει ΜΟΝΟ η B -> το κέρδος θέλει tokens· καταγράφεται ως συμβιβασμός, δεν προχωράει
        χωρίς απόφαση.
    Αν καμία δεν φτάνει 38/58 -> η διάσπαση ΔΕΝ είναι ο δρόμος ούτε στο νέο σετ· επόμενο HippoRAG 2.

ΣΥΝΘΗΚΗ (29/9/2026): 1η απόπειρα με κλειδί ΝΕΟΥ λογαριασμού — «models/gemini-2.5-flash is no longer
available to new users» -> --model gemini-3.8-flash. Δύο εμπόδια: σκέφτεται παρά το thinkingBudget=0
(MAX_OUT, παρακάτω) και το δωρεάν πλάνο δίνει 20 κλήσεις/ημέρα/μοντέλο -> σταμάτησε στις 3. Το
τρέξιμο που ΜΕΤΡΑΕΙ έγινε με το κλειδί του παλιού λογαριασμού και το gemini-2.5-flash = ίδιο μοντέλο
με την παραγωγή, τις παγωμένες μεταφράσεις και τον Αύγουστο. Οι 3 του 3.8 μένουν στο πάγωμα (το
κλειδί του περιέχει το μοντέλο), αχρησιμοποίητες.

ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/decomp_mh40.csv · 40 κλήσεις διάσπασης, 0 της παραγωγής):
    έλεγχοι: σελίδες βάσης = baseline 40/40 ✓ · αναπαραγωγή 1ου περάσματος ✓ · routing: έσπασε 40/40
    στρατηγική        τεκμήρια/58  δύο/29  κάλυψη/113  11 παλιές/32  σελ.  +ερ/−ερ   Δ τεκμήρια (CI 95%)
    βάση                   30          7        66           23       7.5
    C ενιαίο rerank        30          6        65           22       7.5    2/2    +0  [−4, +4]
    D θέσεις/σκέλος        37         14        78           24       7.7    7/1    +7  [+3, +12]
    A ανά σκέλος           35         11        77           23       7.8    8/5    +5  [−1, +11]
    B βάση+4               40         14        84           26      11.2   10/0   +10  [+5, +15]
    ΚΑΝΟΝΑΣ: καμία με ≤ 8 σελίδες δεν φτάνει 38. Η D σταματά στο 37 — περνάει τα άλλα τρία (CI > 0,
    παλιές 24, χάνει 1) — αλλά το όριο ΔΕΝ μετακινείται εκ των υστέρων. Το πετυχαίνει ΜΟΝΟ η B -> κατά
    τον κανόνα ΣΥΜΒΙΒΑΣΜΟΣ: +3.7 σελίδες/ερώτηση (~+4.400 tokens), δεν προχωράει χωρίς απόφαση.
    - Η C (η «σωστή σχεδίαση» του Αυγούστου) ΑΠΟΡΡΙΠΤΕΤΑΙ και στο νέο σετ: 30 -> 30. Ο κριτής με ΟΛΗ
      την ερώτηση ξαναθάβει ό,τι φέρνουν τα σκέλη — ο μηχανισμός του h002, τώρα σε 29 ερωτήσεις.
    - Κέρδος υπάρχει ΜΟΝΟ όταν κάθε σκέλος κρίνεται με το ΔΙΚΟ ΤΟΥ ερώτημα (D/A/B). «Και τα δύο
      papers» 7 -> 14. Σχεδόν όλο στις ελληνικές: el 19 -> 25 (D) / 28 (B) από 44 · en 11 -> 12 από 14.
    - ΔΙΟΡΘΩΝΕΙ ΤΟ TRACE: από τις 21 που χάνονταν ΠΡΙΝ τον κριτή, η D σώζει 3 και η B 5· τα κέρδη
      είναι κυρίως ο «κριτής» (4/5). 18/28 δεν μπαίνουν ούτε στην A (~4 πρώτες σελίδες ΚΑΘΕ σκέλους):
      το σκέλος βρίσκει το paper, όχι τη σελίδα (το «7/9 έχουν άλλη σελίδα του ίδιου paper» του trace).
      Σε μερικές η μετάφραση είναι ήδη λίστα λέξεων (m046, m064, m071) και τα σκέλη βγαίνουν γενικά
      («Cloud computing performance») -> ΤΑΒΑΝΙ της διάσπασης ~40/58.
    ΠΡΟΒΛΕΨΗ: βάση ✓ · routing ✓ · C ✗ (30, όχι 34-38) · D ✗ (37, όχι 40-46) · D χάνει ≤ 2 ✓ (1) ·
      A ✗ (35) · A παλιές < 23 ✗ (23) · B ≥ D ✓ · B ~11 σελ. ✓ · κλήση διάσπασης 0.6-1.5 s ✓ (1.09).
      Έξι στις δέκα — ΥΠΕΡΕΚΤΙΜΗΣΑ ΤΗ ΔΙΑΣΠΑΣΗ ΠΑΝΤΟΥ.
    ΧΡΟΝΟΣ ΑΝΑΚΤΗΣΗΣ: ΔΕΝ μετρήθηκε έγκυρα — μηχάνημα με 2.77 GB για το Docker, βάση 12.2 s διάμεσος
      έναντι 0.97 s στο baseline. Οι δύο προβλέψεις χρόνου μένουν ανοιχτές.
    Ο λόγος (β) του Αυγούστου ΙΣΧΥΕΙ ακόμα για ΟΛΕΣ: +1 κλήση Gemini σε ΚΑΘΕ ερώτηση (~1.1 s).

    docker compose exec backend python evaluation/probe_decomp_mh40.py --limit 2   # δοκιμή, 2 ανά σετ
    docker compose exec backend python evaluation/probe_decomp_mh40.py             # ~40 κλήσεις, 1η φορά
"""
import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import bootstrap_ci
import eval_near_ooc as E
import scoreboard as SB

import gemini_rest

SETS = [("mh_old", "golden_multihop_new.jsonl"), ("mh_new", "golden_multihop_v2.jsonl")]
BASE = os.path.join(SB.OUT_DIR, "baseline.json")
TRACE = os.path.join(SB.RUNS, "trace_multihop.csv")
DEC_PATH = os.path.join(SB.RUNS, "decomp_mh40_gemini.json")
OUT = os.path.join(SB.RUNS, "decomp_mh40.csv")
STRATS = ["base", "C", "D", "A", "B"]
LABEL = {"base": "βάση", "C": "C ενιαίο rerank", "D": "D θέσεις/σκέλος",
         "A": "A ανά σκέλος", "B": "B βάση+4"}
# Το 3.8-flash ΣΚΕΦΤΕΤΑΙ παρά το thinkingBudget=0: q051 -> MAX_TOKENS στα 256 (default του
# generate_once)· διαγνωστική κλήση 29/9: 362 thought tokens για δύο γραμμές. Όριο = και καπάκι κόστους.
MAX_OUT = 1024
TARGET_EV = 38            # κανόνας απόφασης (docstring)
OLD_COV_MIN = 23
MAX_LOST_Q = 2


def key(meta: dict) -> str:
    return f"{meta.get('file_name')}:{meta.get('page')}"


def round_robin(page_lists: list, limit: int, skip=()) -> list:
    """Εναλλάξ από κάθε λίστα σελίδων, χωρίς διπλές (και χωρίς όσες είναι στο skip), έως `limit`."""
    out, seen = [], set(skip)
    for tier in itertools.zip_longest(*page_lists):
        for pg in tier:
            if pg is None or key(pg[1]) in seen:
                continue
            seen.add(key(pg[1]))
            out.append(pg)
            if len(out) >= limit:
                return out
    return out


class Pipe:
    """Το 1ο πέρασμα του search_documents με τα στάδια εκτεθειμένα — ΙΔΙΕΣ κλήσεις, ΙΔΙΑ ορίσματα."""

    def __init__(self, ai):
        assert not ai.USE_BGE_SPARSE, "το probe αναπαράγει το search_documents ΧΩΡΙΣ 3ο σκέλος"
        self.ai = ai
        where = ai._build_where(None, E.TEST_USER)
        self.allowed = ai.collection.get(where=where, include=[])["ids"]
        self.idx = ai._get_bm25_index()
        self.dm = ai._get_dense_matrix()

    def candidates(self, q: str) -> list:
        a = self.ai
        d = a._dense_exact_ids(self.dm, q, self.allowed, min(a.DENSE_CANDIDATES, len(self.allowed)))
        s = a._bm25_sparse_ids(self.idx, q, self.allowed, a.DENSE_CANDIDATES)
        return a._rrf_fuse(d, s, self.idx["ids"], self.idx["texts"], self.idx["metas"],
                           k=60, top_n=a.RERANK_CANDIDATES, pos=self.idx["pos"])

    def rerank(self, q: str, items: list) -> list:
        scores = self.ai.reranker.predict([[q, it[1]] for it in items],
                                          batch_size=self.ai.RERANK_BATCH_SIZE)
        return sorted(zip((float(x) for x in scores), [it[1] for it in items],
                          [it[2] for it in items]), key=lambda x: x[0], reverse=True)

    def pages(self, sf: list, n: int) -> list:
        return self.ai._expand_to_pages(sf[:self.ai.EXPAND_INPUT], n, E.TEST_USER)

    def passes(self, sf: list) -> bool:
        return bool(sf) and sf[0][0] >= self.ai.MIN_RERANK_SCORE


def strategies(pipe: Pipe, q: str, subs: list, base: list) -> tuple[dict, dict]:
    """-> ({στρατηγική: σελίδες}, {χρόνοι ms}). Καλείται ΜΟΝΟ όταν το αρχικό πέρασε τον φύλακα."""
    mp = pipe.ai.MAX_PAGES
    orig_c = pipe.candidates(q)                      # ήδη στο cache διανυσμάτων από τη βάση

    t0 = time.perf_counter()
    leg_c = [pipe.candidates(s) for s in subs]
    t_cand = time.perf_counter() - t0
    t0 = time.perf_counter()
    leg_sf = [pipe.rerank(s, c) for s, c in zip(subs, leg_c)]
    t_leg_rr = time.perf_counter() - t0
    leg_pages = [pipe.pages(sf, mp) for sf in leg_sf if pipe.passes(sf)]

    # C: κοινή κλίμακα — ΟΛΑ τα υποψήφια κρίνονται με το ΑΡΧΙΚΟ ερώτημα
    pool, seen = [], set()
    for it in itertools.chain(orig_c, *leg_c):
        if it[1] not in seen:
            seen.add(it[1])
            pool.append(it)
    t0 = time.perf_counter()
    sf_c = pipe.rerank(q, pool)
    t_pool = time.perf_counter() - t0
    out = {"C": pipe.pages(sf_c, mp)}

    if leg_pages:
        reserved = round_robin([lp[:2] for lp in leg_pages], mp // 2)
        taken = {key(m) for _t, m in reserved}
        out["D"] = reserved + [pg for pg in base if key(pg[1]) not in taken][:mp - len(reserved)]
        out["A"] = round_robin(leg_pages, mp)
        out["B"] = base + round_robin(leg_pages, 4, skip={key(m) for _t, m in base})
    else:                                            # κανένα σκέλος πάνω από τον φύλακα -> η βάση
        out["D"] = out["A"] = out["B"] = base
    ms = {"legs_ms": round(1000 * (t_cand + t_leg_rr)), "c_ms": round(1000 * (t_cand + t_pool)),
          "pool": len(pool), "legs_passed": len(leg_pages)}
    return out, ms


def ev_hits(t: dict, pages: list) -> int | None:
    if not t.get("evidence_pages"):
        return None
    got = {key(m) for _x, m in pages}
    return sum(ep in got for ep in t["evidence_pages"])


def cell(row: dict, s: str) -> str:
    ev = "" if row[f"{s}_ev"] is None else f"{row[f'{s}_ev']}/{row['n_ev']} "
    return f"{s} {ev}κ{row[f'{s}_cov']}/{row['n_kw']} σ{row[f'{s}_n']}"


def save_dec(dec: dict) -> None:
    tmp = DEC_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DEC_PATH)


def parse_subs(raw: str, q: str) -> list:
    """Ίδια ανάγνωση με το probe_decomposition (Αύγουστος)."""
    subs = [ln.strip(" -•\t") for ln in raw.strip().split("\n") if ln.strip()]
    subs = [s for s in subs if len(s) > 3][:3]
    return subs or [q]


async def run(args) -> int:
    real_gen = gemini_rest.generate_once            # ΠΡΙΝ από κατασκόπους/πάγωμα: μόνο για τη διάσπαση
    with open(BASE, encoding="utf-8") as f:
        base_rows = {(r["set"], r["id"]): r for r in json.load(f)["rows"]}
    with open(TRACE, encoding="utf-8-sig") as f:
        trace = {(r["id"], r["evidence"]): r["stage"] for r in csv.DictReader(f)}
    tests = [(sk, t) for sk, fn in SETS for t in E.load_jsonl(os.path.join(E.HERE, fn))]
    if args.limit:
        tests = [x for sk, _fn in SETS for x in [y for y in tests if y[0] == sk][:args.limit]]
    dec = {}
    if os.path.exists(DEC_PATH):
        with open(DEC_PATH, encoding="utf-8") as f:
            dec = json.load(f)

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != SB.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {SB.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()          # μεταφράσεις ΜΟΝΟ από το πάγωμα, όπως το scoreboard
    ai_core.ENABLE_CORRECTIVE = True
    ai_core._save_query_emb_cache = lambda: None

    async def no_gemini(prompt, **kw):
        raise RuntimeError("νέα κλήση Gemini της παραγωγής — δεν επιτρέπεται σε αυτό το script")

    gemini_rest.generate_once = no_gemini
    frozen = SB.Frozen(SB.FROZEN_PATH, fresh=False)
    frozen.install()
    from probe_decomposition import DEFAULT_PROMPT, covered  # ΜΕΤΑ το open_store: ίδιο ai_core

    pipe = Pipe(ai_core)
    model = args.model or ai_core.GEMINI_MODEL
    rows, mismatch, repro_mismatch, errors, new_calls = [], [], [], [], 0
    print(f"\n{len(tests)} multi_hop · βάση + {len(STRATS) - 1} στρατηγικές · διάσπαση με {model}\n")
    for sk, t in tests:
        t0 = time.perf_counter()
        base, _b1, _b2, _rw, outcome = await E.retrieve(ai_core, t["question"])
        base_sec = time.perf_counter() - t0
        if ";".join(key(m) for _x, m in base) != base_rows[(sk, t["id"])]["pages"]:
            mismatch.append(t["id"])
        q = ai_core._translation_cache.get(t["question"], t["question"])

        prompt = DEFAULT_PROMPT.format(query=q)
        # κλειδί = μοντέλο + prompt: άλλο μοντέλο -> νέα κλήση, ποτέ απάντηση άλλου μοντέλου
        h = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
        if h not in dec:
            t1 = time.perf_counter()
            try:
                raw = await real_gen(prompt, model=model, api_key=ai_core.GEMINI_API_KEY,
                                     max_output_tokens=MAX_OUT)
            except gemini_rest._RateLimited as e:
                # 429 ΜΕΤΑ τις 5 επαναλήψεις = εξαντλημένο όριο (δωρεάν πλάνο: 20/ημέρα/μοντέλο).
                # Οι επόμενες θα αποτύχουν ΟΛΕΣ — σταμάτα αντί να τις σφυροκοπάς.
                errors.append(f"{t['id']}: όριο Gemini εξαντλήθηκε ({str(e)[:120]})")
                print(f"  {t['id']} ΟΡΙΟ GEMINI — σταματάω· έτοιμες {len(rows)}/{len(tests)}", flush=True)
                break
            except Exception as e:
                errors.append(f"{t['id']}: {type(e).__name__}: {e}"[:300])
                print(f"  {t['id']} διάσπαση ΑΠΕΤΥΧΕ: {errors[-1]}", flush=True)
                continue
            dec[h] = {"id": t["id"], "model": model, "query": q, "out": raw,
                      "ms": round(1000 * (time.perf_counter() - t1))}
            new_calls += 1
            save_dec(dec)                            # ό,τι πληρώθηκε σώζεται ΑΜΕΣΩΣ, όχι στο τέλος
        subs = parse_subs(dec[h]["out"], q)

        res, ms = {s: base for s in STRATS}, {}
        if outcome == "passed_gate" and len(subs) > 1:
            if ";".join(key(m) for _x, m in pipe.pages(pipe.rerank(q, pipe.candidates(q)),
                                                         ai_core.MAX_PAGES)) \
                    != ";".join(key(m) for _x, m in base):
                repro_mismatch.append(t["id"])
            got, ms = strategies(pipe, q, subs, base)
            res.update(got)

        row = {"id": t["id"], "set": sk, "lang": SB.lang_of(t, t["question"]), "outcome": outcome,
               "n_sub": len(subs), "n_kw": len(t["keywords"]), "n_ev": len(t.get("evidence_pages") or []),
               "base_sec": round(base_sec, 2), "gemini_ms": dec[h]["ms"],
               "legs_ms": ms.get("legs_ms"), "c_ms": ms.get("c_ms"), "pool": ms.get("pool"),
               "legs_passed": ms.get("legs_passed")}
        for s in STRATS:
            p = res[s]
            row[f"{s}_ev"] = ev_hits(t, p)
            row[f"{s}_cov"] = covered(p, t["keywords"])
            row[f"{s}_both"] = SB.both_docs(t, p)
            row[f"{s}_n"] = len(p)
            row[f"{s}_pages"] = ";".join(key(m) for _x, m in p)
        row["arrived_new"] = {s: [ep for ep in (t.get("evidence_pages") or [])
                                  if ep in row[f"{s}_pages"].split(";")
                                  and ep not in row["base_pages"].split(";")] for s in STRATS}
        row["lost_base"] = {s: [ep for ep in (t.get("evidence_pages") or [])
                                if ep in row["base_pages"].split(";")
                                and ep not in row[f"{s}_pages"].split(";")] for s in STRATS}
        row["evidence"] = t.get("evidence_pages") or []
        row["query"], row["subqueries"] = q, " || ".join(subs)
        rows.append(row)

        print(f"  {t['id']:<5} {row['lang']} {len(subs)} {SB.SHORT[outcome]:<10}| "
              + " | ".join(cell(row, s) for s in STRATS), flush=True)

    if errors or frozen.errors or frozen.misses:
        print(f"\n!! σφάλματα διάσπασης {len(errors)} · κλήσεις έξω από το πάγωμα "
              f"{len(frozen.errors) + frozen.misses} — ΤΟ ΤΡΕΞΙΜΟ ΔΕΝ ΜΕΤΡΑΕΙ "
              f"(ξανατρέξε: ό,τι πληρώθηκε είναι σωσμένο)")
        return 1
    report(rows, trace, mismatch, repro_mismatch, new_calls, frozen.hits)

    flat = []
    for r in rows:
        r = dict(r)
        r["evidence"] = ";".join(r["evidence"])
        for k in ("arrived_new", "lost_base"):
            for s in STRATS[1:]:
                r[f"{s}_{k}"] = ";".join(r[k][s])
            del r[k]
        flat.append(r)
    if not args.limit:
        with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        print(f"Σώθηκε: {OUT}")
    return 1 if mismatch or repro_mismatch else 0


def report(rows, trace, mismatch, repro_mismatch, new_calls, hits) -> None:
    new = [r for r in rows if r["set"] == "mh_new"]
    old = [r for r in rows if r["set"] == "mh_old"]
    n_ev = sum(r["n_ev"] for r in new)
    print("\n" + "#" * 100)
    print(f"ΕΛΕΓΧΟΣ 1: σελίδες βάσης = baseline.json σε {len(rows) - len(mismatch)}/{len(rows)}"
          + (f"  !! διαφέρουν: {mismatch}" if mismatch else "  ✓"))
    print("ΕΛΕΓΧΟΣ 2: αναπαραγωγή 1ου περάσματος = βάση"
          + (f"  !! διαφέρει σε {repro_mismatch}" if repro_mismatch else "  ✓"))
    print(f"Gemini: {new_calls} νέες κλήσεις διάσπασης · 0 της παραγωγής ({hits} από το πάγωμα)")

    split = [r for r in rows if r["n_sub"] > 1]
    print(f"\nROUTING: έσπασε {len(split)}/{len(rows)} · υπο-ερωτήματα {dict(Counter(r['n_sub'] for r in rows))}"
          + (f" · ΔΕΝ έσπασε: {[r['id'] for r in rows if r['n_sub'] == 1]}" if len(split) < len(rows) else ""))
    frozen_out = [r["id"] for r in rows if r["outcome"] != "passed_gate"]
    if frozen_out:
        print(f"ΑΜΕΤΑΒΛΗΤΕΣ (ο φύλακας έκοψε το αρχικό): {frozen_out}")

    print(f"\n{'στρατηγική':<18}{'τεκμήρια/' + str(n_ev):>13}{'δύο/' + str(len(new)):>9}"
          f"{'κάλυψη/' + str(sum(r['n_kw'] for r in rows)):>12}{'11 παλιές/' + str(sum(r['n_kw'] for r in old)):>14}"
          f"{'σελ. μ.ό.':>10}{'+ερωτ.':>8}{'−ερωτ.':>8}   Δ τεκμήρια (CI 95%)")
    verdicts = {}
    for s in STRATS:
        ev = sum(r[f"{s}_ev"] for r in new)
        both = sum(r[f"{s}_both"] or 0 for r in new)
        cov = sum(r[f"{s}_cov"] for r in rows)
        cov_old = sum(r[f"{s}_cov"] for r in old)
        npg = statistics.mean(r[f"{s}_n"] for r in rows)
        won = sum(r[f"{s}_ev"] > r["base_ev"] for r in new)
        lost = sum(bool(r["lost_base"][s]) for r in new)
        line = (f"{LABEL[s]:<18}{ev:>13}{both:>9}{cov:>12}{cov_old:>14}{npg:>10.1f}"
                f"{won:>8}{lost:>8}")
        if s != "base":
            m, lo, hi = bootstrap_ci.paired_ci([(r["base_ev"], r[f"{s}_ev"]) for r in new], 10000, 42)
            line += f"   {m * len(new):+.1f} [{lo * len(new):+.1f}, {hi * len(new):+.1f}]"
            verdicts[s] = (ev >= TARGET_EV, lo > 0, cov_old >= OLD_COV_MIN, lost <= MAX_LOST_Q,
                           max(r[f"{s}_n"] for r in rows) <= 8)
        print(line)
    print("(Δ τεκμήρια: σελίδες σε σύνολο 29 ερωτήσεων, ζευγαρωτό bootstrap 10.000, seed 42)")

    print("\nΠΟΙΟ ΣΤΑΔΙΟ ΑΠΩΛΕΙΑΣ ΣΩΖΕΙ (από το trace_multihop — πόσες από τις χαμένες της βάσης φτάνουν):")
    from_trace = {(r["id"], ep): trace.get((r["id"], ep), "?")
                  for r in new for ep in r["evidence"] if ep not in r["base_pages"].split(";")}
    stages = Counter(from_trace.values())
    order = ["κανένα σκέλος", "συγχώνευση", "κριτής", "όριο σελίδων", "σιωπή", "?"]
    print(f"  {'':<18}" + "".join(f"{st:>16}" for st in order if stages[st]))
    print(f"  {'χάθηκαν στη βάση':<18}" + "".join(f"{stages[st]:>16}" for st in order if stages[st]))
    for s in STRATS[1:]:
        got = Counter(from_trace[(r["id"], ep)] for r in new for ep in r["arrived_new"][s])
        print(f"  {LABEL[s]:<18}" + "".join(f"{got[st]:>16}" for st in order if stages[st]))

    def med(k, unit="ms"):
        v = [r[k] for r in rows if r.get(k) is not None]
        return f"{statistics.median(v):.0f} {unit}" if v else "-"
    # C: άνω όριο — ξαναβαθμολογεί και τα 15 του αρχικού, που η παραγωγή έχει ήδη
    print(f"\nΧΡΟΝΟΣ (διάμεσος): βάση {statistics.median(r['base_sec'] for r in rows):.2f} s · "
          f"επιπλέον: σκέλη (D/A/B) {med('legs_ms')} · C {med('c_ms')} "
          f"(pool {med('pool', 'ζεύγη')}) · κλήση διάσπασης {med('gemini_ms')} "
          f"(με τις επαναλήψεις του gemini_rest — σε υπερφόρτωση ΔΕΝ είναι ο χρόνος της παραγωγής)")

    print("\nΚΑΝΟΝΑΣ (≥38/58 · CI>0 · παλιές ≥23/32 · χάνει ≤2 · ≤8 σελίδες):")
    for s, v in verdicts.items():
        print(f"  {LABEL[s]:<18}" + " ".join("✓" if x else "✗" for x in v)
              + ("   ΠΕΡΝΑΕΙ" if all(v) else ""))

    show = [r for r in new if any(r["lost_base"][s] for s in STRATS[1:])]
    if show:
        print("\nΕΧΑΣΑΝ σελίδα-τεκμήριο της βάσης:")
        for r in show:
            print(f"  {r['id']}: " + " · ".join(f"{s} {r['lost_base'][s]}" for s in STRATS[1:]
                                               if r["lost_base"][s]))
    print("#" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N ερωτήσεις ανά σετ (δοκιμή· δεν σώζει CSV)")
    ap.add_argument("--model", default=None, help="μοντέλο ΜΟΝΟ της διάσπασης (default: της παραγωγής)")
    args = ap.parse_args()
    try:
        fd = os.open(SB.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {SB.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(SB.LOCK)


if __name__ == "__main__":
    sys.exit(main())
