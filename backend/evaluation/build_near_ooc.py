"""ΚΟΝΤΙΝΕΣ ερωτήσεις εκτός σώματος (near out-of-corpus) — Φάση 0.1 του πλάνου.

ΓΙΑΤΙ:
    Το golden_set_50 έχει 5 out_of_corpus, σχεδόν όλες ΜΑΚΡΙΝΕΣ (quantum, Bitcoin,
    κλινικές δοκιμές, GPU tensor cores· μόνο το q048 GDPR είναι κοντινό). 0/5
    διαρροές -> άνω όριο 45% (rule of three). Και το test-domains (25/9) έδειξε ότι
    διαρρέουν οι ΚΟΝΤΙΝΕΣ (Βεζούβιος, στατίνες: +1.4…+3.9 logits μετά την
    αναδιατύπωση). Με ~50 κοντινές και 0 διαρροές το άνω όριο πέφτει στο ~6%.

ΜΕΘΟΔΟΣ (η ιδέα του UAEval4RAG, ACL 2025 — δική μας υλοποίηση):
    1. ΓΕΝΝΗΣΗ ανά paper: τίτλος + χαρακτηριστικοί όροι (corpus_glossary, η ΙΔΙΑ
       στατιστική μέθοδος με την παραγωγή) + η αρχή του κειμένου. Το Gemini γράφει
       ερωτήσεις που ΑΚΟΥΓΟΝΤΑΙ του πεδίου αλλά ζητούν κάτι που το σώμα δεν λέει.
       Τέσσερις τύποι, ώστε μια διαρροή να αναλύεται ανά μηχανισμό:
         entity — συγκεκριμένο όνομα (νόμος, προϊόν, σύστημα) που δεν αναφέρεται
         number — αριθμός/τιμή/ημερομηνία που δεν αναφέρεται
         later  — εξελίξεις ΜΕΤΑ το νεότερο paper του σώματος (LATER_AFTER)
         combo  — δύο θέματα που το σώμα ΚΑΛΥΠΤΕΙ, σε ερώτηση που ΔΕΝ απαντά
    2. ΕΠΑΛΗΘΕΥΣΗ με ΟΛΟΚΛΗΡΟ το σώμα στο context (~140k tokens ανά παρτίδα):
       «απαντιέται; αν ναι, αυτούσιο απόσπασμα». Το απόσπασμα ελέγχεται
       ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΑ ότι υπάρχει στο κείμενο. Κρατιούνται ΜΟΝΟ όσες πήραν «no».
    3. ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ στο runs/near_ooc_review.json:
         "drop":  {cid: λόγος}  — ο άνθρωπος ΜΟΝΟ πετάει, δεν ξαναβάζει ό,τι
                                   έκοψε η μηχανή (στην αμφιβολία, πετιέται)
         "notes": {cid: σχόλιο} — περνάει στο jsonl ως review_note· καταγράφει
                                   ΠΡΙΝ την αξιολόγηση ποιες περιμένουμε δύσκολες

ΔΥΟ ΣΧΕΔΙΑΣΤΙΚΕΣ ΑΠΟΦΑΣΕΙΣ ΑΠΟ ΜΕΤΡΗΣΕΙΣ ΤΟΥ PROJECT:
    • Η επαλήθευση γίνεται ΠΑΝΤΑ με την ΑΓΓΛΙΚΗ εκδοχή της ερώτησης. Το
      probe_grounding_verdict έδειξε ότι το LLM κρίνει πως αγγλικό κείμενο «δεν
      απαντά» σε ελληνική ερώτηση ακόμα κι όταν την απαντά (6 ΝΑΙ/14 ΟΧΙ έναντι
      34/2). Εδώ αυτό θα έβαζε ΑΠΑΝΤΗΣΙΜΕΣ ελληνικές ερωτήσεις στο σετ.
    • ΔΕΝ απαιτείται ο όρος-στόχος να έχει df=0. Θα κρατούσε μόνο τον τύπο entity
      (GDPR) και θα έκοβε τον ρεαλιστικότερο combo, όπου όλες οι λέξεις υπάρχουν.
      Το df καταγράφεται ως ΕΤΙΚΕΤΑ (focus_df) για ανάλυση, όχι ως φίλτρο.

ΤΟ ΣΦΑΛΜΑ ΠΟΥ ΜΕΤΡΑΕΙ: μια ερώτηση που ΑΠΑΝΤΙΕΤΑΙ αλλά μπαίνει στο σετ θα μετρηθεί
αργότερα ως «διαρροή» ενώ δεν είναι. Γι' αυτό τρία στρώματα (LLM + απόσπασμα +
άνθρωπος) και το «partial» πετιέται κι αυτό. Και στην αξιολόγηση, κάθε «διαρροή»
ελέγχεται ΠΡΩΤΑ για σφάλμα σετ.

ΜΗ ΝΤΕΤΕΡΜΙΝΙΣΜΟΣ: η γέννηση τρέχει με temperature 0.7 για ποικιλία. Δεν πειράζει:
τεχνούργημα είναι το ΑΡΧΕΙΟ, που μετά την επαλήθευση μένει σταθερό. Οι υποψήφιες
σώζονται στο runs/, ώστε μια αποτυχία στην επαλήθευση να ΜΗΝ ξανακαίει γέννηση.
Τα id είναι ΣΤΑΘΕΡΑ (c057 -> n057): ένας νέος γύρος ή ένα «drop» δεν αλλάζει τα
id των υπόλοιπων, άρα τα CSV της αξιολόγησης μένουν συγκρίσιμα.

ΣΩΜΑ: τα 7 PDF του test_papers/cloud/ — το σώμα πάνω στο οποίο μετρήθηκαν τα νούμερα,
ΟΧΙ το τρέχον ευρετήριο της παραγωγής (λείπει το 1706.03178, υπάρχουν 3 επιπλέον).

1ος ΓΥΡΟΣ (25/9/2026): 56 υποψήφιες · απαντήσιμες 26 (46%) · μηχανή κράτησε 30 ·
άνθρωπος πέταξε 6 -> 24. ΠΡΟΒΛΕΨΗ «~1 στις 5 απαντήσιμη» -> 46%: ΛΑΘΟΣ. Και το
smoke (n=8, 75% απαντήσιμες) έδειχνε ~14 κρατημένες έναντι 30 — n=8 ΔΕΝ εκτιμά
απόδοση. Δύο μετρημένες αδυναμίες του 1ου γύρου, διορθωμένες στον 2ο:
    • το «later» αγκυρωνόταν στη χρονιά του ΔΙΚΟΥ ΤΟΥ paper («μετά το 2017») και το
      απαντούσαν τα νεότερα papers: 6/14 partial, όλες με αγκύρωση πριν το 2019.
    • το ίδιο θέμα γεννιόταν από πολλά papers («stateful … since X» ΠΕΝΤΕ φορές),
      επειδή όλα βλέπουν την ίδια επισκόπηση -> λίστα αποφυγής στο prompt.
Αυτόματο φίλτρο σχεδόν-διπλότυπων ΔΟΚΙΜΑΣΤΗΚΕ και ΔΕΝ μπήκε: η επικάλυψη λέξεων
(Jaccard) δίνει 0.18 σε πραγματικό διπλότυπο (c013/c030, ασφάλεια «μετά το 2019»
/ «μετά το 2020») και 0.64 σε ΔΙΑΦΟΡΕΤΙΚΕΣ ερωτήσεις. Κανένα κατώφλι δεν χωρίζει·
τυπώνεται μόνο η πλησιέστερη ως ΥΠΟΔΕΙΞΗ για τον άνθρωπο.

ΠΡΟΒΛΕΨΗ 2ου ΓΥΡΟΥ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: απαντήσιμες 30-45% (το later πέφτει
κάτω από 1/3)· μετά τον ανθρώπινο έλεγχο 20-28 νέες -> σύνολο 44-52.
2ος ΓΥΡΟΣ: απαντήσιμες 38% ✓ · later 1/14 ✓ · νέες 19 ✗ -> ΤΕΛΙΚΟ 42 (η c039 του 1ου
γύρου έφυγε: η σχεδόν ίδια c095 κρίθηκε partial, σωστά). Η λίστα αποφυγής σταματά
τις ίδιες λέξη προς λέξη, ΟΧΙ τις παραφράσεις (ασφάλεια ×5, latency ×5 στον γύρο 2).

ΚΟΣΤΟΣ ανά γύρο: 7 κλήσεις γέννησης (~5k tokens) + ~5 κλήσεις επαλήθευσης (~145k
tokens) ≈ 0.8M tokens εισόδου. Καμία επαφή με την παραγωγή ή τη ChromaDB.

    # δοκιμή ΑΚΡΟ-ΕΩΣ-ΑΚΡΟ σε 1 paper (1 γέννηση + 1 επαλήθευση), γράφει *_smoke:
    docker compose exec backend python evaluation/build_near_ooc.py --papers 1
    # 1ος γύρος:
    docker compose exec backend python evaluation/build_near_ooc.py
    # ΝΕΟΣ γύρος: επαληθεύει ΜΟΝΟ τις νέες, κρατάει τις αποφάσεις των παλιών:
    docker compose exec backend python evaluation/build_near_ooc.py --append
    # ΜΗΔΕΝ κλήσεις: ξαναγράφει το jsonl με τον τρέχοντα ανθρώπινο έλεγχο:
    docker compose exec backend python evaluation/build_near_ooc.py --apply-review
    # ξανά επαλήθευση ΟΛΩΝ, με τις ήδη σωσμένες υποψήφιες:
    docker compose exec backend python evaluation/build_near_ooc.py --reuse-candidates
"""
import argparse
import asyncio
import glob
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymupdf

import corpus_glossary
import gemini_rest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(HERE, "test_papers", "cloud")
REVIEW_PATH = os.path.join(HERE, "runs", "near_ooc_review.json")
GOLDEN_50 = os.path.join(HERE, "golden_set_50.jsonl")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

TYPES = ("entity", "number", "later", "combo")
# Το νεότερο paper του σώματος είναι το 1902.03383 (Φεβ. 2019). Ένα «later»
# αγκυρωμένο σε παλιότερη χρονιά το απαντούν τα νεότερα papers (1ος γύρος: 6/14).
LATER_AFTER = 2019
# Οι 5 out_of_corpus του golden_set_50: το νέο σετ μένει ΞΕΝΟ προς αυτές.
EXISTING_FOCI = {"quantum", "bitcoin", "gdpr", "clinical", "tensor"}
REFERENCE = {
    "el": "Τα έγγραφα δεν απαντούν σε αυτό· το σύστημα πρέπει να δηλώσει ότι δεν βρίσκει "
          "την απάντηση στα διαθέσιμα έγγραφα.",
    "en": "The provided documents do not answer this; the system should state that it "
          "cannot find the answer in the provided documents.",
}

GEN_PROMPT = """You are building a TEST SET for a question-answering system over this \
collection of research papers:

{overview}

Below is the beginning of ONE of these papers:
--- {file} ---
{excerpt}
---

Write {n} questions that a real user of this system might plausibly ask: they must sound \
like they belong to the SAME FIELD as these papers, but ask for something that this paper \
- and, as far as you can tell from the overview, the other papers - does NOT contain.
Use these types, {per_type} of each:
- "entity": asks about a specific named thing of the field (a law, product, company, \
system, standard, person) that the papers do not mention.
- "number": asks for a specific number, price, date or measurement that the papers do \
not report.
- "later": asks about developments AFTER {later_after}. The newest paper in the \
collection is from early {later_after}, so developments after an OLDER year are already \
covered by the newer papers - always anchor to {later_after} or later.
- "combo": combines two topics that the papers DO discuss into a question they do not \
answer.

These questions ALREADY exist in the test set. Do NOT repeat them, paraphrase them, or ask \
for the same missing thing in other words:
{avoid}

Rules:
- Natural user phrasing, one sentence. Do not ask whether the paper mentions something.
- Nothing obviously off-topic (no cryptocurrencies, medicine, quantum computing, sports).
- Exactly half of the questions in Greek (lang "el"), half in English (lang "en"), \
mixed across types.
- For every question give "question_en": its English version (identical to "question" \
when lang is "en").
- "focus": the specific thing asked for that the papers lack, in English, 1-4 words.

Return ONLY a JSON array, no prose:
[{{"type": "...", "lang": "el|en", "question": "...", "question_en": "...", "focus": "..."}}]
"""

VERIFY_PROMPT = """Below is the COMPLETE text of a collection of research papers, page by \
page. After it there is a list of questions.

For EACH question decide whether the papers answer it:
- "yes": a careful reader could give a correct, specific answer using ONLY these papers.
- "partial": the papers answer part of the question, or a closely related question.
- "no": the papers do not answer it. General background on the same topic that does not \
answer the actual question counts as "no".
For "yes" and "partial" copy ONE supporting sentence VERBATIM from the papers into "quote" \
and give "where" as "file p.N". For "no" leave both empty.

=== PAPERS ===
{corpus}
=== END OF PAPERS ===

QUESTIONS:
{questions}

Return ONLY a JSON array, one object per question, in the same order:
[{{"id": "...", "verdict": "yes|partial|no", "quote": "...", "where": "..."}}]
"""


def normalize_pdf_text(text: str) -> str:
    """ΑΝΤΙΓΡΑΦΟ του ai_core._normalize_pdf_text — εδώ για να μη φορτώνονται τα
    μοντέλα (40-60 s) σε script που δεν τα χρειάζεται. Ίδια εξαγωγή με το ingest."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"([a-zA-Zα-ωΑ-Ω])-\n([a-zα-ω])", r"\1\2", text)


def el_tokenize(text: str) -> list[str]:
    """ΑΝΤΙΓΡΑΦΟ του ai_core.el_tokenize — ο ίδιος tokenizer με το BM25."""
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text, flags=re.UNICODE)


def squash(text: str) -> str:
    """Κανονική μορφή για σύγκριση: ίδια tokens, ενιαία κενά, χωρίς στίξη.
    Έτσι ο έλεγχος αποσπάσματος δεν σκοντάφτει σε αλλαγές γραμμής ή παύλες."""
    return " " + " ".join(el_tokenize(text)) + " "


def content_words(text: str) -> set[str]:
    return {w for w in el_tokenize(text) if len(w) > 2 and w not in corpus_glossary._STOP}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def first_line(text: str) -> str:
    for line in text.splitlines():
        if len(line.strip()) >= 15:
            return line.strip()
    return "(χωρίς τίτλο)"


def load_papers() -> list[dict]:
    papers = []
    for path in sorted(glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))):
        with pymupdf.open(path) as doc:
            pages = [normalize_pdf_text(p.get_text() or "") for p in doc]
            title = ((doc.metadata or {}).get("title") or "").strip()
        papers.append({"file": os.path.basename(path), "title": title or first_line(pages[0]),
                       "pages": pages, "terms": corpus_glossary.document_terms(pages)})
    return papers


def load_review() -> dict:
    """{"drop": {cid: λόγος}, "notes": {cid: σχόλιο}} — λείπει το αρχείο = κανένας έλεγχος."""
    if not os.path.exists(REVIEW_PATH):
        return {"drop": {}, "notes": {}}
    with open(REVIEW_PATH, encoding="utf-8") as f:
        r = json.load(f)
    return {"drop": r.get("drop", {}), "notes": r.get("notes", {})}


def existing_ooc_questions() -> list[str]:
    with open(GOLDEN_50, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["question"] for r in rows if r.get("category") == "out_of_corpus"]


def parse_json_array(raw: str) -> list:
    """Το μοντέλο συχνά τυλίγει το JSON σε ```json … ```· κρατάμε από '[' ως ']'."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"δεν βρέθηκε JSON array στην απάντηση: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


async def generate(papers: list[dict], n: int, avoid: list[str]) -> list[dict]:
    overview = "\n".join(f"- {p['title']} ({p['file']}) — key terms: {', '.join(p['terms'])}"
                         for p in papers)
    avoid_txt = "\n".join(f"- {q}" for q in avoid) or "(none)"
    cands = []
    for p in papers:
        prompt = GEN_PROMPT.format(overview=overview, file=p["file"],
                                   excerpt="\n".join(p["pages"])[:6000],
                                   n=n, per_type=max(1, n // len(TYPES)),
                                   later_after=LATER_AFTER, avoid=avoid_txt)
        raw = await gemini_rest.generate_once(prompt, model=MODEL, api_key=API_KEY,
                                              temperature=0.7, max_output_tokens=4096,
                                              thinking_budget=0)
        items = parse_json_array(raw)
        print(f"  {p['file']:<22} {len(items)} υποψήφιες", flush=True)
        for it in items:
            it["source_doc"] = p["file"]
            cands.append(it)
    return cands


async def verify(cands: list[dict], corpus: str, batch: int, delay: float) -> dict:
    verdicts = {}
    for k in range(0, len(cands), batch):
        chunk = cands[k:k + batch]
        qs = "\n".join(f"{c['cid']}: {c['question_en']}" for c in chunk)
        # thinking ΜΕΣΑ στο maxOutputTokens στο Gemini 2.5 -> αρκετό περιθώριο,
        # αλλιώς η απάντηση βγαίνει κομμένη ή κενή.
        raw = await gemini_rest.generate_once(VERIFY_PROMPT.format(corpus=corpus, questions=qs),
                                              model=MODEL, api_key=API_KEY, temperature=0.0,
                                              max_output_tokens=16384, thinking_budget=1024)
        for v in parse_json_array(raw):
            verdicts[str(v.get("id", "")).strip()] = v
        print(f"  επαλήθευση {k + len(chunk)}/{len(cands)}", flush=True)
        if k + batch < len(cands):
            await asyncio.sleep(delay)   # περιθώριο για το per-minute όριο tokens
    return verdicts


def decide(c: dict, v: dict | None, corpus_sq: str, seen: set) -> tuple[str, str]:
    """(απόφαση, λόγος). ΣΥΝΤΗΡΗΤΙΚΟ: στην αμφιβολία, η ερώτηση ΠΕΤΙΕΤΑΙ — μια
    απαντήσιμη ερώτηση στο σετ θα μετρούσε ψεύτικη διαρροή."""
    words = set(el_tokenize(f"{c.get('focus', '')} {c.get('question_en', '')}"))
    if words & EXISTING_FOCI:
        return "drop", "επικαλύπτει τις 5 υπάρχουσες ooc"
    key = squash(c["question_en"])
    if key in seen:
        return "drop", "διπλότυπο"
    seen.add(key)
    if v is None:
        return "drop", "λείπει ετυμηγορία"
    verdict = str(v.get("verdict", "")).lower()
    if verdict == "no":
        return "keep", ""
    quote = str(v.get("quote") or "")
    quote_ok = len(el_tokenize(quote)) >= 5 and squash(quote) in corpus_sq
    return "drop", f"{verdict} ({v.get('where', '?')}, απόσπασμα {'OK' if quote_ok else 'ΔΕΝ βρέθηκε'})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="υποψήφιες ανά paper (πολλαπλάσιο του 4)")
    ap.add_argument("--papers", type=int, default=0, help="μόνο τα πρώτα N papers (δοκιμή)")
    ap.add_argument("--batch", type=int, default=12, help="ερωτήσεις ανά κλήση επαλήθευσης")
    ap.add_argument("--delay", type=float, default=30.0, help="s ανάμεσα στις επαληθεύσεις")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--reuse-candidates", action="store_true",
                      help="ξανά επαλήθευση ΟΛΩΝ των σωσμένων υποψηφίων, χωρίς γέννηση")
    mode.add_argument("--append", action="store_true",
                      help="νέος γύρος: γέννηση + επαλήθευση ΜΟΝΟ των νέων")
    mode.add_argument("--apply-review", action="store_true",
                      help="ΜΗΔΕΝ κλήσεις: μόνο ο ανθρώπινος έλεγχος -> jsonl")
    args = ap.parse_args()

    if not API_KEY and not args.apply_review:
        print("!! Λείπει GEMINI_API_KEY στο περιβάλλον — σταματάω")
        return 1
    papers = load_papers()
    if len(papers) != 7:
        print(f"!! Περίμενα 7 PDF στο {PAPERS_DIR}, βρήκα {len(papers)} — σταματάω")
        return 1

    suffix = "_smoke" if args.papers else ""
    cand_path = os.path.join(HERE, "runs", f"near_ooc_candidates{suffix}.json")
    out_path = os.path.join(HERE, f"golden_near_ooc{suffix}.jsonl")

    # Το σώμα της επαλήθευσης είναι ΠΑΝΤΑ ολόκληρο (και στη δοκιμή): μια ερώτηση
    # γραμμένη από το paper A μπορεί να την απαντά το paper B.
    corpus = "\n\n".join(f"[{p['file']} p.{i + 1}]\n{t}"
                         for p in papers for i, t in enumerate(p["pages"]))
    corpus_sq = squash(corpus)
    page_sq = [squash(t) for p in papers for t in p["pages"]]
    print(f"Σώμα: {len(papers)} papers · {len(page_sq)} σελίδες · {len(corpus):,} χαρ "
          f"≈ {len(corpus) / 4.53:,.0f} tokens ανά κλήση επαλήθευσης")

    old: list[dict] = []
    if args.reuse_candidates or args.append or args.apply_review:
        if not os.path.exists(cand_path):
            print(f"!! Δεν υπάρχει {cand_path} — τρέξε πρώτα χωρίς σημαία")
            return 1
        with open(cand_path, encoding="utf-8") as f:
            old = json.load(f)
        for c in old:
            c.setdefault("round", 1)   # ο 1ος γύρος γράφτηκε πριν υπάρξει το πεδίο
        print(f"Φορτώθηκαν {len(old)} υποψήφιες από {cand_path}")

    if args.reuse_candidates or args.apply_review:
        cands = old
        to_verify = old if args.reuse_candidates else []
    else:
        gen_papers = papers[:args.papers] if args.papers else papers
        # Αποφυγή: ΟΛΕΣ οι προηγούμενες υποψήφιες (και οι πεταμένες — μια
        # απαντήσιμη δεν θέλουμε να ξαναγεννηθεί) + οι 5 ooc του golden_set_50.
        avoid = [c["question_en"] for c in old] + existing_ooc_questions()
        print(f"\nΓέννηση ({len(gen_papers)} papers × {args.n}, λίστα αποφυγής {len(avoid)}):")
        raw_cands = asyncio.run(generate(gen_papers, args.n, avoid))
        next_id = max((int(c["cid"][1:]) for c in old), default=0) + 1
        rnd = max((c["round"] for c in old), default=0) + 1
        new = []
        for it in raw_cands:
            ok = (it.get("type") in TYPES and it.get("lang") in REFERENCE
                  and all(str(it.get(k) or "").strip()
                          for k in ("question", "question_en", "focus")))
            if ok:
                it["cid"] = f"c{next_id + len(new):03d}"
                it["round"] = rnd
                new.append(it)
        print(f"  έγκυρη μορφή: {len(new)}/{len(raw_cands)}")
        cands = old + new
        to_verify = new
        os.makedirs(os.path.dirname(cand_path), exist_ok=True)
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False, indent=1)

    verdicts = {}
    if to_verify:
        print(f"\nΕπαλήθευση {len(to_verify)} με ΟΛΟΚΛΗΡΟ το σώμα (παρτίδες των {args.batch}):")
        verdicts = asyncio.run(verify(to_verify, corpus, args.batch, args.delay))

    verify_ids = {c["cid"] for c in to_verify}
    seen: set = set()
    for c in cands:
        if c["cid"] in verify_ids:
            v = verdicts.get(c["cid"])
            c["verdict"] = (v or {}).get("verdict", "-")
            c["decision"], c["reason"] = decide(c, v, corpus_sq, seen)
        else:
            # Αποφασισμένη σε προηγούμενο γύρο: η απόφαση μένει, αλλά μετράει στα διπλότυπα.
            seen.add(squash(c["question_en"]))
        focus_sq = squash(c.get("focus", ""))
        c["focus_df"] = sum(focus_sq in s for s in page_sq) if focus_sq.strip() else -1
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)

    review = load_review()
    unknown = sorted((set(review["drop"]) | set(review["notes"])) - {c["cid"] for c in cands})
    if unknown:
        print(f"!! Ο έλεγχος αναφέρει cid που δεν υπάρχουν: {unknown}")
    machine_kept = [c for c in cands if c["decision"] == "keep"]
    kept = [c for c in machine_kept if c["cid"] not in review["drop"]]
    words = {c["cid"]: content_words(c["question_en"]) for c in kept}

    shown = [c for c in cands if c["cid"] in verify_ids] or machine_kept
    print(f"\n{'cid':<6}{'τύπος':<8}{'γλ':<4}{'ετυμ.':<9}{'df':>4}  απόφαση / ερώτηση")
    for c in shown:
        if c["decision"] != "keep":
            mark = f"πετιέται: {c['reason']}"
        elif c["cid"] in review["drop"]:
            mark = f"πετιέται (άνθρωπος): {review['drop'][c['cid']]}"
        else:
            # ΥΠΟΔΕΙΞΗ, όχι φίλτρο: η επικάλυψη λέξεων δεν χωρίζει τα διπλότυπα (βλ. docstring).
            near = max(((jaccard(words[c["cid"]], words[o["cid"]]), o["cid"])
                        for o in kept if o["cid"] != c["cid"]), default=(0.0, "-"))
            mark = f"ΚΡΑΤΙΕΤΑΙ   (πλησιέστερη {near[1]} {near[0]:.2f})"
        print(f"{c['cid']:<6}{c['type']:<8}{c['lang']:<4}{c['verdict']:<9}{c['focus_df']:>4}  {mark}\n"
              f"{'':<31}{c['question'][:95]}")

    with open(out_path, "w", encoding="utf-8") as f:
        for c in kept:
            row = {
                "id": "n" + c["cid"][1:], "question": c["question"],
                "keywords": [c["focus"].lower()], "reference_answer": REFERENCE[c["lang"]],
                "category": "out_of_corpus", "near_type": c["type"], "lang": c["lang"],
                "question_en": c["question_en"], "focus": c["focus"],
                "focus_df": c["focus_df"], "source_doc": c["source_doc"], "round": c["round"],
            }
            if c["cid"] in review["notes"]:
                row["review_note"] = review["notes"][c["cid"]]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "#" * 78)
    for r in sorted({c["round"] for c in cands}):
        rc = [c for c in cands if c["round"] == r]
        ans = sum(c["verdict"] in ("yes", "partial") for c in rc)
        later = [c for c in rc if c["type"] == "later"]
        later_ans = sum(c["verdict"] in ("yes", "partial") for c in later)
        print(f"γύρος {r}: υποψήφιες {len(rc)} · απαντήσιμες {ans} ({ans / max(1, len(rc)):.0%})"
              f" · later απαντήσιμες {later_ans}/{len(later)}"
              f" · μηχανή κράτησε {sum(c['decision'] == 'keep' for c in rc)}"
              f" · μετά τον άνθρωπο {sum(c['round'] == r for c in kept)}")
    print(f"ΣΥΝΟΛΟ ΣΤΟ ΣΕΤ: {len(kept)}")
    print("  ανά τύπο:   " + " · ".join(f"{t} {sum(c['type'] == t for c in kept)}" for t in TYPES))
    print("  ανά γλώσσα: " + " · ".join(f"{g} {sum(c['lang'] == g for c in kept)}"
                                         for g in REFERENCE))
    print("ΠΡΟΒΛΕΨΗ 2ου γύρου (γραμμένη πριν): απαντήσιμες 30-45%, later < 1/3, "
          "20-28 νέες μετά τον άνθρωπο -> σύνολο 44-52.")
    print(f"\nΓράφτηκε: {out_path}\nΥποψήφιες + ετυμηγορίες: {cand_path}")
    print("ΕΠΟΜΕΝΟ: διάβασε τις νέες ΚΡΑΤΗΜΕΝΕΣ· όποια κρίνεις απαντήσιμη ή διπλότυπη,\n"
          f"πρόσθεσέ τη στο \"drop\" του {REVIEW_PATH} και τρέξε --apply-review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
