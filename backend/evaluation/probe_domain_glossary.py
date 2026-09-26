"""Συγκρίνει ΤΡΕΙΣ αυτόματες μεθόδους παραγωγής του corpus glossary (που τροφοδοτεί
το translate-then-retrieve στο optimize_query) — ΧΩΡΙΣ να αγγίξει το corpus
παραγωγής (418 chunks).

ΤΟ ΕΡΩΤΗΜΑ: ο σημερινός descriptor είναι ΚΑΘΟΛΙΚΟΣ, φτιαγμένος ΜΙΑ φορά από CLI
(build_corpus_descriptor.py). Απορρίφθηκε ως σχεδίαση: ο χρήστης δεν τρέχει CLI σε
κάθε upload, και ένας descriptor δεν μπορεί να είναι δύο πεδία μαζί. Η εναλλακτική:
εξαγωγή ορολογίας ΑΝΑ ΕΓΓΡΑΦΟ στο ingest. Δύο υποψήφιες μέθοδοι:
  CA = ΣΤΑΤΙΣΤΙΚΗ (μηδέν κόστος, ντετερμινιστική): οι πιο συχνοί όροι του εγγράφου
       (tf, df>=2, χωρίς stopwords/αριθμούς) — «το λεξιλόγιο που ΟΝΤΩΣ χρησιμοποιεί».
  CB = LLM ΑΝΑ ΕΓΓΡΑΦΟ (Gemini Flash στο ingest): ίδιο prompt με τον CLI descriptor,
       αλλά ανά paper· domain + terms.
  CC = ΣΤΑΤΙΣΤΙΚΗ ΚΑΘΑΡΙΣΜΕΝΗ: CA μείον boilerplate (όρος σε ΣΧΕΔΟΝ ΚΑΘΕ σελίδα =
       header/footer, π.χ. το όνομα του περιοδικού «cureus») και μείον διπλές
       θέσεις για ενικό/πληθυντικό («patient» + «patients»).
Και μια συνθήκη ελέγχου:
  C0 = ΧΩΡΙΣ glossary (domain γενικό, terms κενά) — η σημερινή fallback συμπεριφορά.

ΓΙΑΤΙ ΟΧΙ σύγκριση με τον σημερινό καθολικό descriptor: είναι ήδη απορριφθείσα
σχεδίαση. Συγκρίνουμε τις ΝΕΕΣ μεθόδους μεταξύ τους (+ C0 για «βοηθάει καθόλου;»).

ΔΥΟ ΜΕΤΡΙΚΕΣ — η δεύτερη είναι ΚΡΙΤΗΡΙΟ ΑΠΟΚΛΕΙΣΜΟΥ, όχι βαθμολογία:
  1. keyword coverage (in-corpus), ΙΔΙΟΣ ορισμός με το eval_engine (keyword ως
     substring, lowercase, σε ΟΠΟΙΑΔΗΠΟΤΕ ανακτηθείσα σελίδα). Ντετερμινιστικό —
     το glossary ελέγχει τη ΜΕΤΑΦΡΑΣΗ, άρα ό,τι φτάνει στον reranker.
  2. ΔΙΑΡΡΟΕΣ out_of_corpus: σιωπά το gate σε ερώτηση που ΔΕΝ απαντιέται;
     ΓΙΑΤΙ ΜΠΗΚΕ: το glossary σπρώχνει το ερώτημα προς το λεξιλόγιο του corpus —
     ΑΚΡΙΒΩΣ ο μηχανισμός που απέρριψε 4 προηγούμενες ιδέες (query enrichment v1/v2,
     disambiguation-by-retrieval: 5 διαρροές). Το gate δίνει το 100% της αξιοπιστίας
     (θετική σκάλα: +0.000 MRR, 5/5 ooc), άρα +5 coverage με τίμημα μία διαρροή
     είναι ΚΑΘΑΡΗ ΖΗΜΙΑ. Οι «near» ooc (ασπιρίνη/στατίνες/Βεζούβιος — λέξεις που
     ΔΕΝ υπάρχουν πουθενά, αλλά στο ΙΔΙΟ πεδίο) είναι το σκληρό τεστ· οι «far»
     (Bitcoin/κβαντικοί) το εύκολο.
Η γέννηση απάντησης ΔΕΝ μετριέται εδώ (μη ντετερμινιστική, ανθρώπινος έλεγχος χωριστά).

ΑΠΟΜΟΝΩΣΗ (κανένα άγγιγμα στην παραγωγή):
  • ξεχωριστός PersistentClient σε /tmp -> το production chroma.sqlite3 δεν
    ανοίγει ποτέ για γράψιμο.
  • ξεχωριστή διεργασία (docker compose exec) -> τα monkeypatch στα module-globals
    δεν αγγίζουν τον uvicorn server.
  • _save_translation_cache -> no-op ΚΑΙ _translation_cache καθαρίζεται ανά
    συνθήκη -> δεν μολύνεται το production translations.json, και κάθε συνθήκη
    ξαναμεταφράζει με το ΔΙΚΟ της glossary (το cache έχει κλειδί την ΕΡΩΤΗΣΗ).

    docker compose exec backend python evaluation/probe_domain_glossary.py \
        --papers evaluation/test_papers/cureus-0015-00000046486.pdf \
                 evaluation/test_papers/s41598-017-03833-3.pdf \
        --golden evaluation/golden_test_domains.jsonl \
        --csv evaluation/runs/domain_glossary.csv
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from collections import Counter

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from probe_doc_terms import gloss_for as specific_gloss  # ΕΙΔΙΚΟΙ όροι + domain
from suggest_keywords import STOP

import ai_core
import gemini_rest

TEST_USER = 999_999          # εικονικός ιδιοκτήτης των test chunks (authz)
TEST_DB = "/tmp/test_domains_chroma"  # noqa: S108  μέσα στο container, εφήμερο
GENERIC_DOMAIN = "scientific research papers"

# Ίδιο prompt με τον build_corpus_descriptor.py — αλλά ΑΝΑ ΕΓΓΡΑΦΟ (CB).
_EXTRACT_PROMPT = (
    "You are analyzing a scientific document to build a translation glossary for "
    "an English-only search engine. Below are excerpts from ONE document.\n"
    "1) In ONE short line, name the domain/field of this document.\n"
    "2) List up to 15 STANDARD ENGLISH TECHNICAL TERMS that a search over this "
    "document would rely on — the exact vocabulary the document uses.\n"
    "Output EXACTLY this format, nothing else:\n"
    "DOMAIN: <one line>\n"
    "TERMS: term1, term2, term3, ...\n\n"
    "--- DOCUMENT EXCERPTS ---\n{sample}"
)


def _tokens(text: str):
    """Ίδιος tokenizer με το suggest_keywords.terms_of, αλλά κρατάει counts."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return [w for w in words if w not in STOP]


def statistical_terms(page_texts: list[str], top: int = 15) -> list[str]:
    """CA: οι πιο συχνοί όροι του εγγράφου, df>=2 (όχι one-off), κατά tf.

    ΓΙΑΤΙ df>=2: ένας όρος σε μία μόνο σελίδα μπορεί να είναι τυπογραφικό/όνομα·
    η ορολογία που «χρησιμοποιεί το έγγραφο» επαναλαμβάνεται. ΓΙΑΤΙ κατά tf και
    όχι κατά σπανιότητα (όπως το keyword-proxy): εδώ θέλουμε το ΧΑΡΑΚΤΗΡΙΣΤΙΚΟ
    λεξιλόγιο, όχι διακριτικούς όρους μιας σελίδας.
    """
    tf = Counter()
    df = Counter()
    for txt in page_texts:
        toks = _tokens(txt)
        tf.update(toks)
        df.update(set(toks))
    cand = [(c, w) for w, c in tf.items() if df[w] >= 2]
    cand.sort(key=lambda x: (-x[0], x[1]))
    return [w for _c, w in cand[:top]]


def cleaned_terms(page_texts: list[str], top: int = 15,
                  boilerplate_ratio: float = 0.9) -> list[str]:
    """CC: σαν CA, αλλά χωρίς boilerplate και χωρίς διπλά ενικού/πληθυντικού.

    ΤΙ ΔΙΟΡΘΩΝΕΙ (μετρημένο στο 1ο τρέξιμο): οι top-15 του CA στο καρδιολογικό
    paper ήταν «heart, failure, patients, therapies, management, clinical,
    emerging, CUREUS, healthcare, PATIENT, cardiac, hfref, medicine, stem, risk».
    Το «cureus» είναι το όνομα του ΠΕΡΙΟΔΙΚΟΥ (footer κάθε σελίδας) και το
    patient/patients πιάνει δύο θέσεις για την ίδια λέξη -> ~1/3 των θέσεων
    είναι θόρυβος που ο μεταφραστής βλέπει ισότιμα με το «hfref».

    ΓΙΑΤΙ ΟΧΙ χειροποίητη λίστα «ακαδημαϊκών» λέξεων: θα έκοβε τα «rate»,
    «model», «level», «value» — που εδώ είναι ΟΥΣΙΩΔΗ («effusion rate»,
    «magmastatic model»). Το boilerplate ορίζεται ΔΟΜΙΚΑ (σε σχεδόν κάθε
    σελίδα), όχι σημασιολογικά -> δεν χρειάζεται να μαντέψουμε πεδίο.
    """
    n_pages = len(page_texts)
    tf, df = Counter(), Counter()
    for txt in page_texts:
        toks = _tokens(txt)
        tf.update(toks)
        df.update(set(toks))

    # 1. Boilerplate: εμφανίζεται σε (σχεδόν) ΚΑΘΕ σελίδα. Ενεργό μόνο με
    # αρκετές σελίδες — σε 3σέλιδο έγγραφο το «σε κάθε σελίδα» δεν σημαίνει
    # footer, σημαίνει θέμα.
    boiler = set()
    if n_pages >= 6:
        boiler = {w for w, d in df.items() if d >= boilerplate_ratio * n_pages}

    cand = [(c, w) for w, c in tf.items() if df[w] >= 2 and w not in boiler]
    cand.sort(key=lambda x: (-x[0], x[1]))

    # 2. Ενικός/πληθυντικός: κράτα ΜΟΝΟ την πρώτη μορφή που συναντάς (η
    # συχνότερη, αφού η λίστα είναι ταξινομημένη) και άθροισε τη 2η σε αυτήν.
    out, seen_roots = [], set()
    for _c, w in cand:
        root = w[:-2] if w.endswith("ies") else w.rstrip("s")
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(w)
        if len(out) >= top:
            break
    return out


async def llm_terms(page_texts: list[str], model: str) -> tuple[str, str]:
    """CB: domain + terms από το Gemini, ανά έγγραφο (ίδιο prompt με τον CLI)."""
    step = max(1, len(page_texts) // 15)
    sample = "\n---\n".join(t[:600] for t in page_texts[::step][:15])
    raw = (await gemini_rest.generate_once(
        _EXTRACT_PROMPT.format(sample=sample), model=model,
        api_key=ai_core.GEMINI_API_KEY,
        thinking_budget=1024, max_output_tokens=2048)).strip()
    domain, terms = "", ""
    for line in raw.splitlines():
        if line.upper().startswith("DOMAIN:"):
            domain = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TERMS:"):
            terms = line.split(":", 1)[1].strip()
    return domain, terms


# --- Η ΟΔΗΓΙΑ του glossary, ως μεταβλητή -----------------------------------
# Οι CA/CB/CC άλλαξαν ΠΟΙΟΙ όροι μπαίνουν. Καμία δεν άλλαξε ΠΩΣ ζητούνται —
# και η σημερινή διατύπωση («prefer the vocabulary») είναι εντολή ΜΕΤΑΚΙΝΗΣΗΣ
# του ερωτήματος προς το corpus, δηλαδή ακριβώς ο μηχανισμός του query
# enrichment που έχει ήδη απορριφθεί δύο φορές για διαρροή out_of_corpus.
_G_PROD = "Prefer the vocabulary of this corpus glossary: {terms}. "

# Ουδέτερη: οι ΙΔΙΕΣ λέξεις, χωρίς προστακτική. Απαντά στο «φταίει η εντολή ή
# σκέτη η παρουσία των λέξεων;» — αν διαρρέει κι αυτή, καμία διατύπωση δεν σώζει.
_G_NEUTRAL = "Corpus glossary, for reference: {terms}. "

# Περιοριστική: όρος του glossary ΜΟΝΟ ως μετάφραση λέξης που ΥΠΑΡΧΕΙ στην
# ερώτηση. Στοχεύει ακριβώς τη διαφορά v9/o4: το «κλισίμετρα»->tiltmeter είναι
# ΕΠΙΛΟΓΗ ΜΕΤΑΦΡΑΣΗΣ, ενώ το «Βεζούβιος»->eruption vocabulary είναι ΠΡΟΣΘΗΚΗ.
_G_CONSTRAINED = (
    "Corpus glossary: {terms}. Use a glossary term ONLY as the translation of a "
    "word that is actually present in the question. NEVER add a glossary term "
    "that the question does not mention, and NEVER map a proper noun (a place, "
    "person, product or brand name) onto glossary vocabulary. "
)


async def _optimize_query_variant(query: str, glossary_tmpl: str) -> str:
    """ΑΝΤΙΓΡΑΦΟ του ai_core.optimize_query· ΜΟΝΗ διαφορά: η πρόταση του glossary
    είναι παράμετρος αντί για σταθερή.

    ΓΙΑΤΙ αντίγραφο και όχι αλλαγή στην παραγωγή: δεν αγγίζουμε παραγωγικό κώδικα
    πριν η μέτρηση πει ότι αξίζει. CONTROL: η συνθήκη με το _G_PROD πρέπει να
    αναπαράγει τη γραμμή CC του προηγούμενου τρεξίματος (v9 100%, o4 +0.49).
    """
    if not ai_core._has_greek(query):
        return query
    if query in ai_core._translation_cache:
        return ai_core._translation_cache[query]
    terms = ai_core._CORPUS_TERMS
    glossary = glossary_tmpl.format(terms=terms) if terms else ""
    prompt = (
        "You are a translation assistant for an English-only academic search engine. "
        f"The corpus is about: {ai_core._CORPUS_DOMAIN}. "
        "Translate the user's question to English using the STANDARD TECHNICAL "
        "TERMINOLOGY of that field: map everyday words to the term a document would "
        "actually use, NOT the literal everyday translation. "
        + glossary
        + "Preserve every domain term already present in the question - never drop or "
        "generalize one. Output ONLY a concise English search query with the key "
        "terms. No quotes, no extra text.\n\n"
        f"User question: {query}"
    )
    english = (await gemini_rest.generate_once(
        prompt, model=ai_core.GEMINI_MODEL,
        api_key=ai_core.GEMINI_API_KEY)).strip(" \"'\n")
    ai_core._translation_cache[query] = english
    return english


def load_golden(path: str) -> list[dict]:
    tests = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    return tests


def coverage(keywords: list[str], retrieved_texts: list[str]) -> float:
    """ΙΔΙΟΣ ορισμός με eval_engine.evaluate_retrieval."""
    if not retrieved_texts or not keywords:
        return 0.0
    joined = "\n".join(retrieved_texts).lower()
    found = sum(1 for k in keywords if k.lower() in joined)
    return found / len(keywords) * 100.0


# --- Καταγραφή του best logit ΧΩΡΙΣ αλλαγή συμπεριφοράς --------------------
# Το search_documents δεν επιστρέφει το σκορ του gate. Το πιάνουμε από το ΠΡΩΤΟ
# predict κάθε ερώτησης (το 1ο pass· αν τρέξει corrective, ακολουθεί δεύτερο και
# ΔΕΝ το θέλουμε). Ο spy μόνο διαβάζει — τα scores επιστρέφονται αυτούσια.
_probe_best: dict = {"first": None}


def _install_reranker_spy():
    orig = ai_core.reranker.predict

    def spy(pairs, **kw):
        scores = orig(pairs, **kw)
        if _probe_best["first"] is None and len(scores):
            _probe_best["first"] = float(max(scores))
        return scores

    ai_core.reranker.predict = spy


async def run_condition(name, domain, terms, glossary_tmpl, tests, out_rows):
    """Θέτει glossary + ΟΔΗΓΙΑ, καθαρίζει το translation cache, τρέχει τις μετρήσεις.

    In-corpus -> coverage. Out_of_corpus -> ΣΙΩΠΗ (σωστό) ή ΔΙΑΡΡΟΗ (λάθος).
    Τα δύο ΔΕΝ αθροίζονται: το coverage είναι βαθμολογία, οι διαρροές είναι
    κριτήριο αποκλεισμού.
    """
    ai_core._CORPUS_DOMAIN = domain
    ai_core._CORPUS_TERMS = terms
    ai_core._translation_cache.clear()   # ξανα-μετάφραση με ΤΟ ΔΙΚΟ της glossary
    # **kw: το search_documents περνάει πια domain=/terms= από το assemble() —
    # ΑΓΝΟΟΥΝΤΑΙ, τη μεταβλητή την ορίζει η συνθήκη μέσω των globals.
    ai_core.optimize_query = lambda q, **kw: _optimize_query_variant(q, glossary_tmpl)
    print(f"\n===== {name} =====")
    print(f"  DOMAIN: {domain}")
    print(f"  ΟΔΗΓΙΑ: {glossary_tmpl.split('{terms}')[0]}...{glossary_tmpl.split('{terms}')[-1][:80]}")
    print(f"  TERMS : {terms[:110] + ('...' if len(terms) > 110 else '') if terms else '(κενό)'}")
    per_paper, leaks = {}, []
    for t in tests:
        ooc = t.get("category") == "out_of_corpus"
        _probe_best["first"] = None
        # Πρώτα η μετάφραση ΞΕΧΩΡΙΣΤΑ, για να την καταγράψουμε· το search_documents
        # μετά τη βρίσκει στο cache -> ΙΔΙΑ συμβολοσειρά, ΜΗΔΕΝ επιπλέον κλήση.
        tr = await ai_core.optimize_query(t["question"])
        retrieved = await ai_core.search_documents(
            t["question"], target_filenames=None, user_id=TEST_USER)
        texts = [text for text, _meta in retrieved]
        best = _probe_best["first"]
        best_s = f"{best:+.2f}" if best is not None else "  n/a"
        row = {"condition": name, "id": t["id"], "paper": t.get("paper", "-"),
               "coverage": "", "pages": len(texts), "best_logit": best_s.strip(),
               "verdict": "", "translation": tr}

        if ooc:
            leaked = len(texts) > 0
            if leaked:
                leaks.append(t["id"])
            row["verdict"] = "ΔΙΑΡΡΟΗ" if leaked else "σιωπή"
            out_rows.append(row)
            mark = "!! ΔΙΑΡΡΟΗ" if leaked else "   σιωπή  "
            print(f"    {t['id']:<4} {mark} best {best_s}  [{t.get('distance','')}]"
                  f"  -> {tr[:52]}")
            continue

        cov = coverage(t["keywords"], texts)
        per_paper.setdefault(t["paper"], []).append(cov)
        row["coverage"] = round(cov, 1)
        out_rows.append(row)
        print(f"    {t['id']:<4} cov {cov:5.1f}%  ({len(texts)} σελ)  best {best_s}"
              f"  -> {tr[:44]}")

    for paper, covs in per_paper.items():
        print(f"  -- {paper[:32]:<32} μέση cov {sum(covs)/len(covs):5.1f}%  (n={len(covs)})")
    allc = [r["coverage"] for r in out_rows
            if r["condition"] == name and r["coverage"] != ""]
    mean_cov = sum(allc) / len(allc)
    n_ooc = sum(1 for t in tests if t.get("category") == "out_of_corpus")
    print(f"  == {name}: μέση cov {mean_cov:5.1f}%  ·  ooc σιωπηλά "
          f"{n_ooc - len(leaks)}/{n_ooc}"
          + (f"  ΔΙΑΡΡΟΕΣ: {', '.join(leaks)}" if leaks else ""))
    return mean_cov, leaks


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--llm-model", default=ai_core.GEMINI_MODEL,
                    help="μοντέλο για τη μέθοδο CB (default: το production GEMINI_MODEL)")
    ap.add_argument("--with-llm", action="store_true",
                    help="τρέξε ΚΑΙ τη μέθοδο CB (2 κλήσεις εξαγωγής + 1 συνθήκη)")
    ap.add_argument("--with-specific", action="store_true",
                    help="τρέξε ΚΑΙ τη CG: ΕΙΔΙΚΟΙ όροι ανά έγγραφο (probe_doc_terms) "
                         "με την ΙΔΙΑ περιοριστική οδηγία του CD -> μόνη μεταβλητή "
                         "το ΠΕΡΙΕΧΟΜΕΝΟ των όρων")
    ap.add_argument("--specific-model", default="gemini-pro-latest",
                    help="μοντέλο εξαγωγής για τη CG")
    ap.add_argument("--only", default="",
                    help="ονόματα συνθηκών χωρισμένα με κόμμα (π.χ. CD,CG) — "
                         "παράλειψη των υπολοίπων")
    ap.add_argument("--corrective", action="store_true",
                    help="άφησε τον corrective agent ΑΝΟΙΧΤΟ (default: κλειστός, "
                         "αλλιώς η μέτρηση ΔΕΝ είναι επαναλήψιμη)")
    args = ap.parse_args()

    # --- ΑΠΟΜΟΝΩΣΗ: μη γράφεις ΠΟΤΕ στο production translations.json ---
    ai_core._save_translation_cache = lambda: None

    # ΓΙΑΤΙ κλειστός ο corrective: τρέχει ΜΟΝΟ όταν κόβει το gate, με rewrite από
    # το Gemini -> εισάγει μη επαναλήψιμο σκέλος ακριβώς στις ερωτήσεις που μας
    # ενδιαφέρουν. Στο προηγούμενο τρέξιμο το v3 του CB βγήκε 100% ΕΝΩ το 1ο pass
    # του ήταν -5.61 (κομμένο) -> το «+5 του CB» ήταν του agent, όχι του glossary.
    if not args.corrective:
        ai_core.ENABLE_CORRECTIVE = False
        print("ENABLE_CORRECTIVE = False (μετράμε ΜΟΝΟ το 1ο pass -> επαναλήψιμο)")

    # --- Test collection σε ξεχωριστό store (καθαρό κάθε φορά) ---
    if os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB)
    test_client = chromadb.PersistentClient(path=TEST_DB)
    test_col = test_client.get_or_create_collection(
        name="test_domains",
        embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})

    # Στρέψε ΟΛΟ το ai_core στο test collection. Οι caches (bm25/dense) διαβάζουν
    # το module-global collection -> μετά το swap χτίζονται από το test corpus.
    ai_core.collection = test_col

    print(f"Ingest {len(args.papers)} papers στο απομονωμένο store {TEST_DB} ...")
    for i, path in enumerate(args.papers):
        filename = os.path.basename(path)
        ok = ai_core.ingest_pdf(path, filename, user_id=TEST_USER,
                                is_public=False, doc_id=i)
        print(f"  {filename}: {'OK' if ok else 'ΚΕΝΟ'}")
    ai_core._bump_corpus_version()   # σιγουριά: invalidate caches πριν το 1ο query
    _install_reranker_spy()          # μόνο ΔΙΑΒΑΖΕΙ τα scores — καμία αλλαγή

    # Κείμενα ανά paper (για τα glossaries), από το test collection.
    got = test_col.get(include=["documents", "metadatas"])
    by_paper: dict[str, list[str]] = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        by_paper.setdefault(meta["file_name"], []).append(doc)

    # --- Χτίσιμο των δύο glossaries (ένωση ανά έγγραφο, όπως θα γινόταν στο
    #     query-time assembly όταν είναι στο scope και τα δύο papers) ---
    ca_terms_sets, cc_terms_sets, cb_domains, cb_terms_sets = [], [], [], []
    cg_terms_sets, cg_domains = [], []
    for paper, docs in by_paper.items():
        st = statistical_terms(docs)
        cl = cleaned_terms(docs)
        ca_terms_sets.append(st)
        cc_terms_sets.append(cl)
        print(f"\n[{paper}]  ({len(docs)} chunks)")
        print(f"  CA στατιστικά : {', '.join(st)}")
        print(f"  CC καθαρισμένα: {', '.join(cl)}")
        print(f"     -> CC βγάζει: {', '.join(w for w in st if w not in cl) or '(τίποτα)'}")
        if args.with_llm:
            dom, tm = await llm_terms(docs, args.llm_model)
            cb_domains.append(dom)
            cb_terms_sets.append(tm)
            print(f"  CB llm domain : {dom}")
            print(f"  CB llm terms  : {tm}")
        if args.with_specific:
            g = await specific_gloss(paper, docs, 15, args.specific_model)
            cg_terms_sets.append(g["terms"])
            cg_domains.append(g["domain"])
            print(f"  CG domain     : {g['domain']}")
            print(f"  CG ΕΙΔΙΚΟΙ    : {g['terms']}")

    # (η CA τυπώνεται για σύγκριση αλλά ΔΕΝ τρέχει ξανά: μετρήθηκε ταυτόσημη με CC)
    cc_terms = ", ".join(dict.fromkeys(t for s in cc_terms_sets for t in s))
    cb_terms = ", ".join(dict.fromkeys(
        t.strip() for s in cb_terms_sets for t in s.split(",") if t.strip()))
    cb_domain = "; ".join(d for d in cb_domains if d) or GENERIC_DOMAIN

    # ΤΙ ΜΕΤΑΒΑΛΛΕΤΑΙ ΠΟΥ: C0 = χωρίς όρους (πάτωμα). Οι υπόλοιπες έχουν ΤΟΥΣ
    # ΙΔΙΟΥΣ όρους (CC) και διαφέρουν ΜΟΝΟ στην οδηγία -> ό,τι διαφορά βγει,
    # βγαίνει από τη διατύπωση, όχι από το λεξιλόγιο.
    conds = [
        ("C0_no_glossary", GENERIC_DOMAIN, "", _G_PROD),
        ("CC_prod_instr", GENERIC_DOMAIN, cc_terms, _G_PROD),
        ("CD_constrained", GENERIC_DOMAIN, cc_terms, _G_CONSTRAINED),
        ("CE_neutral", GENERIC_DOMAIN, cc_terms, _G_NEUTRAL),
    ]
    if args.with_llm:
        conds.append(("CB_llm", cb_domain, cb_terms, _G_PROD))
    if args.with_specific:
        # CG: ΙΔΙΟ domain και ΙΔΙΑ οδηγία με το CD — αλλάζει ΜΟΝΟ το είδος των όρων
        # (ειδικοί αντί για στατιστικά συχνοί). Στο κύριο σετ αυτή η αλλαγή
        # επανέφερε το coverage 97.04% -> 98.52% (17/8). Εδώ ελέγχεται αν κρατάει
        # το +10 των νέων πεδίων — αν το χάσει, το κέρδος ήταν των γενικών όρων.
        cg_terms = ", ".join(dict.fromkeys(
            t.strip() for s in cg_terms_sets for t in s.split(",") if t.strip()))
        conds.append(("CG_specific", GENERIC_DOMAIN, cg_terms, _G_CONSTRAINED))

        # CH: ΤΟ ΑΚΡΙΒΕΣ ΣΧΗΜΑ ΤΗΣ ΠΑΡΑΓΩΓΗΣ — ίδιοι όροι με τη CG, αλλά το domain
        # δεν είναι καρφωμένο: είναι η ΕΝΩΣΗ των per-file domains, όπως θα το χτίζει
        # το query-time assembly από τα metadata των εγγράφων που είναι στο scope.
        #
        # ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΗ ΣΥΝΘΗΚΗ ΚΑΙ ΟΧΙ ΕΠΕΚΤΑΣΗ ΤΗΣ CG: εδώ είναι το ΡΙΣΚΟ
        # ΔΙΑΡΡΟΗΣ. Η CG πέρασε με domain «scientific research papers» — ουδέτερο
        # απέναντι στο o4 («Πότε εξερράγη ο Βεζούβιος;»). Το πραγματικό domain θα
        # λέει «Volcanology ...», δηλαδή ΑΚΡΙΒΩΣ το λεξιλόγιο της out_of_corpus
        # ερώτησης. Ο ΙΔΙΟΣ μηχανισμός που διέρρευσε το o4 στις CC/CE (+0.49) και
        # που απέρριψε το query enrichment και το disambiguation-by-retrieval:
        # λεξιλόγιο του corpus μέσα στο ερώτημα -> ο reranker βαθμολογεί το
        # λεξιλόγιο, όχι την ερώτηση.
        #
        # ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: coverage >= 95.0% (όσο η CG) ΚΑΙ
        # out_of_corpus 5/5 σιωπηλά. Οι διαρροές είναι ΑΠΟΚΛΕΙΣΤΙΚΟ κριτήριο — δεν
        # ανταλλάσσονται με coverage. Αν το o4 διαρρεύσει, το domain ΔΕΝ μπαίνει
        # στην παραγωγή ανά αρχείο και μένει καρφωμένο γενικό (η CG είναι το
        # fallback που έχει ήδη περάσει).
        ch_domain = "; ".join(d for d in cg_domains if d) or GENERIC_DOMAIN
        conds.append(("CH_doc_domain", ch_domain, cg_terms, _G_CONSTRAINED))

        # CJ/CK: ΤΟ ΚΕΝΟ ΚΕΛΙ — domain ΑΝΑ ΑΡΧΕΙΟ, ΜΗΔΕΝ terms. Καμία συνθήκη δεν
        # το δοκίμασε: η C0 είχε ΓΕΝΙΚΟ domain χωρίς terms (85%), οι CG/CH ΕΙΔΙΚΟ
        # domain ΜΕ terms (95%). Άρα το +10 χρεώθηκε στα terms χωρίς να ελεγχθεί αν
        # το σωστό domain αρκεί. Αν αρκεί, τα terms — η μόνη επιφάνεια διαρροής —
        # φεύγουν εντελώς αντί να τιθασεύονται με διατύπωση.
        # ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: >= 95.0% ΚΑΙ 5/5 σιωπηλά. Διαφορετικά
        # η υπόθεση πέφτει και το backfill προχωράει με terms.
        conds.append(("CJ_doc_domain_only", ch_domain, "", _G_PROD))
        # CK: ίδια metadata, σχήμα του _FALLBACK_DOMAIN («<είδος> papers on <υποπεδία>»)
        # αντί για σκέτες ετικέτες ενωμένες με «;». Απομονώνει τη διατύπωση.
        _labels = [d for d in cg_domains if d]
        ck_domain = ("scientific papers on " + ", ".join(d.lower() for d in _labels)
                     if _labels else GENERIC_DOMAIN)
        conds.append(("CK_rich_domain", ck_domain, "", _G_PROD))

    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        conds = [c for c in conds if c[0] in keep or c[0].split("_")[0] in keep]
        print(f"\nΜΟΝΟ οι συνθήκες: {[c[0] for c in conds]}")

    tests = load_golden(args.golden)
    rows: list[dict] = []
    print("\n" + "#" * 70)
    results = []
    for name, dom, tm, tmpl in conds:
        m, lk = await run_condition(name, dom, tm, tmpl, tests, rows)
        results.append((name, m, lk))

    m0 = results[0][1]
    n_ooc = sum(1 for t in tests if t.get("category") == "out_of_corpus")
    print("\n" + "#" * 70)
    print(f"ΣΥΝΟΨΗ — coverage (in-corpus) ΚΑΙ διαρροές (out_of_corpus, n={n_ooc})")
    print(f"{'συνθήκη':<18} {'cov':>7}  {'Δ vs C0':>8}   ooc σιωπηλά")
    for name, m, lk in results:
        flag = f"  <-- ΔΙΑΡΡΟΕΣ: {', '.join(lk)}" if lk else ""
        print(f"{name:<18} {m:6.1f}%  {m-m0:+8.1f}   "
              f"{n_ooc - len(lk)}/{n_ooc}{flag}")
    print("\nΤο coverage είναι βαθμολογία· οι διαρροές είναι ΚΡΙΤΗΡΙΟ ΑΠΟΚΛΕΙΣΜΟΥ.")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "id", "paper", "coverage",
                                              "pages", "best_logit", "verdict",
                                              "translation"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nΓράφτηκε: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
