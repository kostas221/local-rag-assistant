"""Τι κάνει το ΣΗΜΕΡΙΝΟ σύστημα στις 42 ΚΟΝΤΙΝΕΣ ερωτήσεις εκτός σώματος; — Φάση 0.1.

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Οι 42 ερωτήσεις του golden_near_ooc.jsonl «ακούγονται» σαν να τις απαντούν τα
    papers (cold start, κόστος Lambda, OpenWhisk...), αλλά ζητούν κάτι που ΔΕΝ
    υπάρχει μέσα τους (Python 3.9 στα 256MB, τιμή του 2023, OpenFaaS...). Μέχρι
    τώρα το «0 διαρροές» στηριζόταν σε 5 ΜΑΚΡΙΝΕΣ ερωτήσεις (Bitcoin, quantum):
    άνω όριο 45%. Εδώ μετράμε τις ΚΟΝΤΙΝΕΣ, που είναι ο πραγματικός κίνδυνος.

ΤΙ ΚΑΝΕΙ, ΓΙΑ ΚΑΘΕ ΕΡΩΤΗΣΗ:
    1. Ο ΠΡΑΓΜΑΤΙΚΟΣ `ai_core.search_documents` (μετάφραση, dense, BM25, RRF,
       reranker, gate, corrective) πάνω σε ΑΠΟΜΟΝΩΜΕΝΟ store με τα 7 cloud PDF.
       Ένας «κατάσκοπος» στον reranker σημειώνει τους βαθμούς — ΜΟΝΟ διαβάζει.
    2. Αν δεν κόπηκε: η ΠΡΑΓΜΑΤΙΚΗ `ai_core.ask_ai` γράφει την απάντηση
       (ίδιο system prompt, ίδιο thinking budget, ίδιες παραπομπές).
    3. Κριτής (Gemini) κατατάσσει την απάντηση — ΟΡΙΣΜΟΣ ΠΡΟΚΑΤΑΧΩΡΗΜΕΝΟΣ στο
       AGENTS.md (25/9/2026), ΠΡΙΝ δούμε απαντήσεις:
         correct   — λέει ρητά ότι το ζητούμενο δεν υπάρχει στα έγγραφα (μπορεί να
                     δώσει και το σχετικό υλικό, ΑΡΚΕΙ να το πει)
         soft_leak — δίνει σχετικό υλικό ΧΩΡΙΣ να πει ότι το ζητούμενο λείπει
         leak      — ισχυρίζεται το στοιχείο που λείπει (αριθμό, ιδιότητα, εξέλιξη)
       Ο κριτής επιστρέφει και την ΠΡΟΤΑΣΗ της απάντησης που τον έπεισε· ελέγχεται
       ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΑ ότι υπάρχει αυτούσια (όπως το απόσπασμα στο build_near_ooc).
       Δεύτερο, ΑΝΕΞΑΡΤΗΤΟ σήμα: τα μοτίβα άρνησης (REFUSAL) του probe_no_gate.
       Κάθε soft_leak/leak και κάθε διαφωνία κριτή-μοτίβων ελέγχεται ΜΕ ΤΟ ΜΑΤΙ.

ΔΥΟ ΕΛΕΓΧΟΙ ΑΞΙΟΠΙΣΤΙΑΣ (μόνο ανάκτηση, μηδέν γέννηση):
    • ΑΡΝΗΤΙΚΟΣ: οι 5 out_of_corpus του golden_set_50 πρέπει να ΚΟΒΟΝΤΑΙ.
    • ΘΕΤΙΚΟΣ: 5 in-corpus του golden_set_50 πρέπει να ΠΕΡΝΑΝΕ. Χωρίς αυτόν, ένα
      άδειο ή χαλασμένο store θα έδινε «42/42 κομμένες = τέλειο». Κανόνας του
      project: όταν ένα σετ δίνει τέλειο σκορ, ύποπτο είναι το σετ.
    Αν αποτύχει οποιοσδήποτε έλεγχος, τα νούμερα ΔΕΝ διαβάζονται.

ΑΠΟΜΟΝΩΣΗ (κανένα άγγιγμα στην παραγωγή):
    • ξεχωριστός PersistentClient στο /tmp του container· χτίζεται μία φορά και
      ξαναχρησιμοποιείται (--rebuild για καινούργιο).
    • ξεχωριστή διεργασία -> οι αλλαγές σε globals δεν αγγίζουν τον uvicorn.
    • το cache μεταφράσεων της παραγωγής ΔΕΝ γράφεται· οι μεταφράσεις των νέων
      ερωτήσεων σώζονται στο runs/near_ooc_translations.json και ξαναφορτώνονται,
      ώστε ένα 2ο τρέξιμο να κάνει ΤΗΝ ΙΔΙΑ ανάκτηση.
    • το domain της μετάφρασης μένει ΟΠΩΣ ΣΤΗΝ ΠΑΡΑΓΩΓΗ (corpus_descriptor.json):
      με γενικό domain το gate κόβει h004/h007 (corrective_cloud, 25/9).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (25/9/2026):
    επίπεδο 1: περνάνε 25-35 από τις 42 (60-83%) — οι κοντινές μοιράζονται λεξιλόγιο
               με τα papers και ο reranker είναι λεξιλογικός.
    επίπεδο 2: από όσες περνάνε, leak 0-3 (υποψήφιες: τιμή «του 2023» από τον
               πίνακα του 2019, τιμή Lambda στο Azure) και soft_leak 15-35%.
    Το ιστορικό μου: 1 σωστή πρόβλεψη στις 5 τελευταίες.

ΑΠΟΤΕΛΕΣΜΑ (25/9/2026, runs/near_ooc_eval.csv — ανάλυση στο AGENTS.md, Φάση 0.1):
    επίπεδο 1: περνάνε 40/42 (πρόβλεψη 25-35 ✗).
    επίπεδο 2: σωστό 38 · ήπια διαρροή 1 (n048, οριακά απαντήσιμη) · διαρροή 1
               (n024, συμπέρασμα «η γλώσσα δεν επηρεάζει το autoscaling»).
               leak 0-3 ✓ · soft_leak 2.5% ✗.
    ⚠️ ο αρνητικός έλεγχος q048 (GDPR) ΠΕΡΑΣΕ μέσω corrective: η αναδιατύπωση του
    Gemini άλλαξε ανάμεσα σε δύο τρεξίματα (-5.71 κομμένο / -3.40 πέρασε). Το store
    είναι ακέραιο (οι άλλοι 9 έλεγχοι ταυτόσημοι) — το εύρημα αφορά τον agent.

ΚΟΣΤΟΣ: ~21 μεταφράσεις + λίγες αναδιατυπώσεις + μία γέννηση (~10k tokens) και μία
κλήση κριτή ανά ερώτηση που περνάει ≈ 0.4M tokens εισόδου. Χρόνος: ~5 λεπτά το
πρώτο χτίσιμο του store (embeddings σε CPU) + ~5-10 λεπτά οι ερωτήσεις.

    # δοκιμή σε 3 ερωτήσεις + ελέγχους, γράφει *_smoke.csv:
    docker compose exec backend python evaluation/eval_near_ooc.py --limit 3
    # πλήρες:
    docker compose exec backend python evaluation/eval_near_ooc.py
    # ΜΟΝΟ ξανά κριτής στις ήδη γραμμένες απαντήσεις (καμία ανάκτηση/γέννηση):
    docker compose exec backend python evaluation/eval_near_ooc.py --judge-only
"""
import argparse
import asyncio
import csv
import glob
import json
import math
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_rest

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR_PATH = os.path.join(HERE, "golden_near_ooc.jsonl")
GOLDEN_50 = os.path.join(HERE, "golden_set_50.jsonl")
PAPERS = os.path.join(HERE, "test_papers", "cloud", "*.pdf")
TRANS_PATH = os.path.join(HERE, "runs", "near_ooc_translations.json")
TEST_DB = "/tmp/eval_near_ooc_chroma"  # noqa: S108  εφήμερο, μέσα στο container
TEST_USER = 999_997          # εικονικός ιδιοκτήτης των test chunks (authz)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")
N_POS_CONTROLS = 5
LABELS = ("correct", "soft_leak", "leak")

# Από το probe_no_gate.py (εκεί είναι τεκμηριωμένο γιατί επιτρέπονται ως 3
# παρεμβαλλόμενες λέξεις). Ανεξάρτητο δεύτερο σήμα — ΔΕΝ αποφασίζει μόνο του.
# Μία διεύρυνση: βρ[έε]θηκ -> βρ[έε]θ, για το «δεν είναι δυνατό να βρεθεί» (smoke, n010).
_W = r"(?:\w+\s+){0,3}"
REFUSAL = re.compile(
    r"δεν\s+" + _W + r"(?:βρ[ίι]σκ|βρ[έε]θ|περι[έε]χ|αναφ[έε]ρ|υπ[άα]ρχ|"
    r"παρ[έε]χ|μπορ|δ[ίι]ν|συζητ|εξετ|σχολι|προσδιορ|τεκμηρι|ανακτ)"
    r"|ουδεμ[ίι]α\s+αναφορ|καμ[ίι]α\s+" + _W + r"(?:αναφορ|πληροφορ|μνε[ίι]α)"
    r"|there\s+(?:is|are|was|were)\s+no\s"
    r"|(?:do|does|did)\s+not\s+" + _W + r"(?:contain|mention|provide|report|"
    r"discuss|specify|include|address|state|say|detail|present)"
    r"|(?:is|are|was|were)\s+not\s+" + _W + r"(?:contained|mentioned|provided|"
    r"reported|discussed|specified|included|addressed|present|available|found)"
    r"|\bno\s+" + _W + r"(?:information|answer|mention|reference|results?|data|"
    r"details?|evidence|benchmark|figures?)\b"
    r"|not\s+(?:explicitly\s+|specifically\s+|directly\s+)?(?:mentioned|discussed|"
    r"stated|specified|provided|reported|addressed|contained|available|present)"
    r"|cannot\s+" + _W + r"(?:find|determine|answer|provide|locate)"
    r"|unable\s+to\s+" + _W + r"(?:find|determine|answer|provide|locate)",
    re.I)

JUDGE_PROMPT = """You are auditing a question-answering system that must answer ONLY \
from a fixed set of research papers.

The QUESTION below was written on purpose so that the papers do NOT contain its answer. \
It was checked by reading the complete papers.
What the papers lack: {focus}
Reviewer note about RELATED material that DOES exist in the papers (may be empty): {note}

QUESTION: {question}
QUESTION IN ENGLISH: {question_en}

SYSTEM ANSWER (may be in Greek; citations like [S3] point to pages):
<<<
{answer}
>>>

Classify the SYSTEM ANSWER with exactly one label:
- "correct": it explicitly says that the documents do not contain the specific thing \
asked. It MAY also give related information from the documents, as long as it clearly \
says that the specific answer is missing.
- "soft_leak": it gives related information but never clearly says that the specific \
thing asked is missing, so a reader would take it as the answer.
- "leak": it states the missing fact itself (a number, a property, a development, a \
comparison) as if it were true - from general knowledge or by extrapolating from the \
documents (for example, applying one product's price to another product, or presenting \
an old figure as a current one).

Return ONLY JSON, no prose:
{{"label": "correct|soft_leak|leak", "evidence": "<ONE sentence copied VERBATIM from \
the SYSTEM ANSWER that decided the label>"}}
"""

FIELDS = ["id", "kind", "near_type", "lang", "question", "question_en", "translation",
          "best1", "best2", "rewrite", "outcome", "n_pages", "files", "answer",
          "refusal_regex", "judge", "judge_evidence", "evidence_found", "review_note"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def squash(text: str) -> str:
    return " " + " ".join(re.findall(r"\w+", (text or "").lower())) + " "


def upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """Άνω όριο Clopper-Pearson (μονόπλευρο). k=0 -> ~3/n (rule of three).
    Διχοτόμηση στο διωνυμικό CDF — χωρίς scipy."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1 - conf

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def parse_json_object(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"δεν βρέθηκε JSON στην απάντηση του κριτή: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


async def judge(row: dict) -> tuple[str, str, bool]:
    prompt = JUDGE_PROMPT.format(
        focus=row["focus"], note=row.get("review_note") or "", question=row["question"],
        question_en=row["question_en"], answer=row["answer"])
    raw = await gemini_rest.generate_once(prompt, model=MODEL, api_key=API_KEY,
                                          temperature=0.0, max_output_tokens=4096,
                                          thinking_budget=1024)
    obj = parse_json_object(raw)
    label = str(obj.get("label", "")).strip().lower()
    if label not in LABELS:
        label = f"?{label}"
    evidence = str(obj.get("evidence") or "")
    found = len(evidence.split()) >= 3 and squash(evidence) in squash(row["answer"])
    return label, evidence, found


# --- Κατάσκοπος στον reranker: ΜΟΝΟ διαβάζει, επιστρέφει αυτούσια -------------------
_calls: list[float] = []
_prompts: list[tuple[str, str]] = []   # (αρχή prompt, απάντηση) ανά κλήση Gemini


def install_spies(ai_core) -> None:
    orig_predict = ai_core.reranker.predict

    def predict_spy(pairs, **kw):
        scores = orig_predict(pairs, **kw)
        if len(scores):
            _calls.append(float(max(scores)))
        return scores

    ai_core.reranker.predict = predict_spy

    orig_gen = gemini_rest.generate_once

    async def gen_spy(prompt, **kw):
        text = await orig_gen(prompt, **kw)
        _prompts.append((prompt[:60], text.strip(" \"'\n")))
        return text

    gemini_rest.generate_once = gen_spy


async def retrieve(ai_core, question: str):
    """(σελίδες, best1, best2, rewrite, outcome) από τον ΠΡΑΓΜΑΤΙΚΟ search_documents."""
    _calls.clear()
    _prompts.clear()
    pages = await ai_core.search_documents(question, None, user_id=TEST_USER)
    rewrite = next((t for p, t in _prompts if p.startswith("The following search query")), "")
    best1 = _calls[0] if _calls else None
    best2 = _calls[1] if len(_calls) > 1 else None
    if not pages:
        outcome = "cut"
    elif best1 is not None and best1 >= ai_core.MIN_RERANK_SCORE:
        outcome = "passed_gate"
    else:
        outcome = "passed_corrective"
    return pages, best1, best2, rewrite, outcome


async def answer_of(ai_core, question: str, pages) -> str:
    out = ""
    async for chunk in ai_core.ask_ai(question, target_filenames=None,
                                      user_id=TEST_USER, precomputed=pages):
        if chunk.get("type") == "text":
            out += chunk.get("data", "")
    return out


def fmt(x) -> str:
    return "" if x is None else f"{x:.2f}"


def open_store(rebuild: bool = False):
    """Φορτώνει το ai_core και το στρέφει στο ΑΠΟΜΟΝΩΜΕΝΟ store (το χτίζει αν λείπει).
    Επιστρέφει (ai_core, σωσμένες μεταφράσεις). Κοινό με το probe_corrective_flip.py."""
    print("Φόρτωση μοντέλων (40-60 s χωρίς έξοδο) — ΜΗΝ το διακόψεις...", flush=True)
    import chromadb

    import ai_core

    ai_core._save_translation_cache = lambda: None   # το cache της παραγωγής ΔΕΝ γράφεται
    fresh = rebuild or not os.path.exists(TEST_DB)
    if rebuild and os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB)
    client = chromadb.PersistentClient(path=TEST_DB)
    col = client.get_or_create_collection(
        name="eval_near_ooc", embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    if fresh or col.count() == 0:
        papers = sorted(glob.glob(PAPERS))
        print(f"Χτίσιμο απομονωμένου store: {len(papers)} papers (~5 λεπτά σε CPU)...",
              flush=True)
        for i, path in enumerate(papers):
            ok = ai_core.ingest_pdf(path, os.path.basename(path), user_id=TEST_USER,
                                    is_public=False, doc_id=i)
            print(f"  {os.path.basename(path)}: {'OK' if ok else 'ΚΕΝΟ'}", flush=True)
    ai_core._bump_corpus_version()
    n_chunks = col.count()
    print(f"Store: {n_chunks} chunks στο {TEST_DB} · domain μετάφρασης: "
          f"{ai_core._CORPUS_DOMAIN!r}")
    install_spies(ai_core)

    saved = {}
    if os.path.exists(TRANS_PATH):
        with open(TRANS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        ai_core._translation_cache.update(saved)
        print(f"Μεταφράσεις από προηγούμενο τρέξιμο: {len(saved)}")
    return ai_core, saved


async def run_pipeline(args, rows_out: list[dict]) -> bool:
    """Ανάκτηση + γέννηση. Επιστρέφει True αν πέρασαν ΚΑΙ οι δύο έλεγχοι."""
    ai_core, saved = open_store(args.rebuild)

    golden = load_jsonl(GOLDEN_50)
    neg = [r for r in golden if r.get("category") == "out_of_corpus"]
    pos = [r for r in golden if r.get("category") != "out_of_corpus"][:N_POS_CONTROLS]
    near = load_jsonl(NEAR_PATH)[:args.limit or None]

    # --- Έλεγχοι αξιοπιστίας: μόνο ανάκτηση ---
    print(f"\n===== ΕΛΕΓΧΟΙ (gate {ai_core.MIN_RERANK_SCORE}, "
          f"corrective {ai_core.CORRECTIVE_MIN_SCORE}) =====")
    controls_ok = True
    for kind, items, want in (("ctrl_neg", neg, "cut"), ("ctrl_pos", pos, "passed")):
        for t in items:
            pages, b1, b2, rw, outcome = await retrieve(ai_core, t["question"])
            ok = outcome.startswith(want)
            controls_ok &= ok
            print(f"  {t['id']:<6} {kind:<9} {outcome:<18} best1 {fmt(b1):>6}  "
                  f"{'OK' if ok else '!! ΑΠΕΤΥΧΕ'}", flush=True)
            rows_out.append({"id": t["id"], "kind": kind, "question": t["question"],
                             "best1": fmt(b1), "best2": fmt(b2), "rewrite": rw,
                             "outcome": outcome, "n_pages": len(pages)})

    # --- Οι κοντινές ooc ---
    print(f"\n===== {len(near)} ΚΟΝΤΙΝΕΣ ooc =====")
    for t in near:
        pages, b1, b2, rw, outcome = await retrieve(ai_core, t["question"])
        translation = ai_core._translation_cache.get(t["question"], "")
        if translation:
            saved[t["question"]] = translation
        answer = await answer_of(ai_core, t["question"], pages) if pages else ""
        files = sorted({m.get("file_name", "?") for _x, m in pages})
        print(f"  {t['id']:<5} {t['near_type']:<7} {t['lang']}  {outcome:<18} "
              f"best1 {fmt(b1):>6}  best2 {fmt(b2):>6}  σελ {len(pages)}", flush=True)
        rows_out.append({
            "id": t["id"], "kind": "near", "near_type": t["near_type"], "lang": t["lang"],
            "question": t["question"], "question_en": t["question_en"],
            "translation": translation, "best1": fmt(b1), "best2": fmt(b2),
            "rewrite": rw, "outcome": outcome, "n_pages": len(pages),
            "files": ";".join(files), "answer": answer,
            "refusal_regex": int(bool(REFUSAL.search(answer))) if answer else "",
            "review_note": t.get("review_note", ""), "focus": t["focus"]})

    os.makedirs(os.path.dirname(TRANS_PATH), exist_ok=True)
    with open(TRANS_PATH, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=1)
    return controls_ok


async def run_judge(rows: list[dict]) -> None:
    todo = [r for r in rows if r.get("kind") == "near" and r.get("answer")]
    print(f"\n===== ΚΡΙΤΗΣ σε {len(todo)} απαντήσεις =====")
    for r in todo:
        try:
            label, evidence, found = await judge(r)
        except Exception as e:   # μία χαλασμένη ετυμηγορία δεν ρίχνει τις υπόλοιπες
            label, evidence, found = "?error", str(e)[:200], False
        r["judge"], r["judge_evidence"], r["evidence_found"] = label, evidence, int(found)
        print(f"  {r['id']:<5} {label:<10} regex-άρνηση {r['refusal_regex']}  "
              f"απόσπασμα {'OK' if found else 'ΔΕΝ βρέθηκε'}", flush=True)


def summarize(rows: list[dict], controls_ok: bool | None) -> None:
    near = [r for r in rows if r.get("kind") == "near"]
    n = len(near)
    print("\n" + "#" * 78)
    if controls_ok is False:
        print("!! ΕΝΑΣ Ή ΠΕΡΙΣΣΟΤΕΡΟΙ ΕΛΕΓΧΟΙ ΑΠΕΤΥΧΑΝ — τα νούμερα ΔΕΝ διαβάζονται.")
    cut = sum(r["outcome"] == "cut" for r in near)
    pg = sum(r["outcome"] == "passed_gate" for r in near)
    pc = sum(r["outcome"] == "passed_corrective" for r in near)
    print(f"ΕΠΙΠΕΔΟ 1 — ο φύλακας ({n} ερωτήσεις)")
    print(f"  κόπηκαν (σιωπή)             {cut:>3}")
    print(f"  πέρασαν από το gate         {pg:>3}")
    print(f"  πέρασαν μέσω corrective     {pc:>3}")

    judged = [r for r in near if r.get("answer")]
    counts = {lab: sum(r.get("judge") == lab for r in judged) for lab in LABELS}
    other = len(judged) - sum(counts.values())
    print(f"ΕΠΙΠΕΔΟ 2 — η απάντηση ({len(judged)} που πέρασαν)")
    for lab in LABELS:
        print(f"  {lab:<12} {counts[lab]:>3}")
    if other:
        print(f"  χωρίς έγκυρη ετυμηγορία {other}")
    for lab, name in (("leak", "ΔΙΑΡΡΟΕΣ (γ)"), ("soft_leak", "ΗΠΙΕΣ ΔΙΑΡΡΟΕΣ (β)")):
        k = counts[lab]
        print(f"{name}: {k}/{n} -> άνω όριο 95%: {upper_bound(k, n):.1%}")

    disagree = [r["id"] for r in judged
                if (r.get("judge") == "correct") != bool(int(r.get("refusal_regex") or 0))]
    print(f"Διαφωνία κριτή / μοτίβων άρνησης: {len(disagree)} {disagree}")
    no_ev = [r["id"] for r in judged if str(r.get("evidence_found")) == "0"]
    if no_ev:
        print(f"Απόσπασμα κριτή που ΔΕΝ βρέθηκε στην απάντηση: {no_ev}")

    for key in ("near_type", "lang"):
        vals = sorted({r[key] for r in near})
        parts = []
        for v in vals:
            sub = [r for r in near if r[key] == v]
            parts.append(f"{v} {sum(r['outcome'] != 'cut' for r in sub)}/{len(sub)} πέρασαν, "
                         f"{sum(r.get('judge') == 'soft_leak' for r in sub)}β "
                         f"{sum(r.get('judge') == 'leak' for r in sub)}γ")
        print(f"  ανά {key}: " + " · ".join(parts))
    print("ΠΡΟΒΛΕΨΗ (γραμμένη πριν): περνάνε 25-35/42 · leak 0-3 · soft_leak 15-35% των "
          "περασμένων.")
    print("ΕΠΟΜΕΝΟ: κάθε soft_leak/leak και κάθε διαφωνία -> ΜΕ ΤΟ ΜΑΤΙ, και κάθε leak "
          "ελέγχεται ΠΡΩΤΑ για σφάλμα σετ.")


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*FIELDS, "focus"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N (δοκιμή)")
    ap.add_argument("--rebuild", action="store_true", help="ξαναχτίζει το απομονωμένο store")
    ap.add_argument("--no-judge", action="store_true", help="χωρίς κριτή (μόνο απαντήσεις)")
    ap.add_argument("--judge-only", action="store_true",
                    help="ΜΟΝΟ κριτής στο υπάρχον CSV — καμία ανάκτηση/γέννηση")
    args = ap.parse_args()
    if not API_KEY:
        print("!! Λείπει GEMINI_API_KEY — σταματάω")
        return 1

    suffix = "_smoke" if args.limit else ""
    csv_path = os.path.join(HERE, "runs", f"near_ooc_eval{suffix}.csv")

    controls_ok = None
    if args.judge_only:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Φορτώθηκαν {len(rows)} γραμμές από {csv_path}")
    else:
        rows = []
        controls_ok = await run_pipeline(args, rows)
        write_csv(csv_path, rows)   # σώζεται ΠΡΙΝ τον κριτή: αποτυχία του δεν καίει γέννηση
    if not args.no_judge:
        await run_judge(rows)
        write_csv(csv_path, rows)
    summarize(rows, controls_ok)
    print(f"\nΓράφτηκε: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
