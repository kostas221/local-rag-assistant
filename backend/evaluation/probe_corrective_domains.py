"""Μπορεί ο corrective agent να δουλέψει σε ΟΠΟΙΟΔΗΠΟΤΕ πεδίο χωρίς να διαρρεύσει;

ΤΟ ΠΡΟΒΛΗΜΑ:
    Το `ai_core._CORRECTIVE_PROMPT` λέει στο Gemini ότι το corpus είναι «cloud
    computing, serverless computing and distributed systems» και ότι «αν η ερώτηση
    είναι ΕΚΤΟΣ του πεδίου, γύρνα την αμετάβλητη». Σε PDF άλλου πεδίου ΚΑΘΕ ερώτηση
    είναι «εκτός» -> ο agent δεν κάνει τίποτα. Η ΙΔΙΑ φράση όμως σήμερα ΠΡΟΣΤΑΤΕΥΕΙ
    τα άλλα πεδία, κατά τύχη: το o4 («Πότε εξερράγη ο Βεζούβιος;») παίρνει −3.46 στο
    1ο πέρασμα, δηλαδή ΠΑΝΩ από το CORRECTIVE_MIN_SCORE (−3.8). Αν ο agent αρχίσει
    να το ξαναγράφει και το 2ο πέρασμα βγει στο ίδιο ύψος, ΠΕΡΝΑΕΙ χωρίς υλικό.
    Δηλαδή το «σβήνω τη λέξη cloud» ανοίγει τρύπα — γι' αυτό μετράμε ΠΡΙΝ.

Η ΥΠΟΘΕΣΗ ΠΟΥ ΕΛΕΓΧΕΤΑΙ — άμυνα που ΔΕΝ χρειάζεται να ξέρει το πεδίο:
    «η αναδιατύπωση πρέπει να ΑΝΕΒΑΣΕΙ τη βαθμολογία» (Δ = best2 − best1 ≥ margin).
    Σε ερώτηση εκτός corpus δεν υπάρχει υλικό να βρεθεί, όσο καλά κι αν ξαναγραφτεί·
    σε ασαφή ερώτηση μέσα στο corpus οι σωστοί όροι φέρνουν το υλικό. Ένδειξη από
    το cloud (runs/corr_v1_control.csv, n=11, ΜΗΔΕΝ κόστος): τα 5 out_of_corpus
    έχουν Δ ∈ {−2.59, −0.24, −0.02, 0.00, 0.00}, τα 4 με σωστό υλικό Δ ≥ +0.88.
    Καθαρό διάστημα (−0.02, +0.88) -> margin = +0.43, το ΜΕΣΟ (ίδια αρχή με το
    gate −2.6 και το corrective −3.8: μεγιστοποιεί το ΧΕΙΡΟΤΕΡΟ περιθώριο).
    Ο κανόνας είναι ΣΧΕΤΙΚΟΣ (η ερώτηση απέναντι στον εαυτό της, στο ίδιο corpus),
    άρα δεν θέλει βαθμονόμηση ανά πεδίο — σε αντίθεση με το απόλυτο −3.8.
    ΟΡΙΟ ΤΗΣ ΕΝΔΕΙΞΗΣ: εκείνες οι αναδιατυπώσεις βγήκαν με το CLOUD prompt.

ΤΡΙΑ PROMPTS (ίδια ουρά, αλλάζει ΜΟΝΟ η περιγραφή του πεδίου):
    A_cloud    το σημερινό της παραγωγής, αυτούσιο (CONTROL).
    B_generic  χωρίς πεδίο: «την ορολογία του πεδίου που αφορά Η ΙΔΙΑ Η ΕΡΩΤΗΣΗ».
               Φεύγει και η φράση «εκτός πεδίου -> αμετάβλητη», γιατί χωρίς πεδίο
               δεν σημαίνει τίποτα — την άμυνα την αναλαμβάνει ο κανόνας Δ.
    C_terms    το σημερινό, με την περιγραφή του πεδίου ΠΑΡΑΓΟΜΕΝΗ από τους όρους
               των in-scope αρχείων (corpus_glossary.assemble — ό,τι ήδη υπάρχει
               στα metadata της παραγωγής). Κρατά τη φράση «εκτός πεδίου».
               ΡΙΣΚΟ ΓΝΩΣΤΟ: οι όροι του corpus μέσα σε prompt αναδιατύπωσης είναι ο
               μηχανισμός που διέρρευσε enrichment/disambiguation.

ΤΙ ΜΕΝΕΙ ΣΤΑΘΕΡΟ — Η ΜΕΤΑΦΡΑΣΗ:
    Με --translations οι μεταφράσεις ΔΕΝ ξαναγίνονται: διαβάζονται από την
    καταγεγραμμένη συνθήκη (default CD_constrained = γενικό domain + όροι ανά
    αρχείο + περιοριστική οδηγία = ΑΚΡΙΒΩΣ η τελική μορφή της παραγωγής χωρίς το
    cloud corpus_descriptor.json). Άρα το 1ο πέρασμα είναι ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ και η
    ΜΟΝΗ μεταβλητή είναι το prompt του agent.
    CONTROL: το best1 κάθε ερώτησης πρέπει να αναπαράγει το best_logit του CSV
    (±0.01). Αν όχι, το απομονωμένο ευρετήριο δεν είναι αυτό που μετρήθηκε και τα
    αποτελέσματα ΔΕΝ διαβάζονται.

ΠΩΣ ΤΡΕΧΕΙ Ο AGENT:
    Καλείται η ΠΡΑΓΜΑΤΙΚΗ `ai_core._corrective_retry` (όχι αντίγραφο), με
    CORRECTIVE_MIN_SCORE = −inf ώστε να επιστρέφει ΠΑΝΤΑ το 2ο πέρασμα· τα
    κατώφλια (σημερινό −3.8, και −3.8 + Δ≥margin) εφαρμόζονται OFFLINE στα ίδια
    νούμερα. Η αναδιατύπωση καταγράφεται τυλίγοντας το gemini_rest.generate_once
    (μόνο διαβάζει — επιστρέφει αυτούσια την απάντηση).

ΑΠΟΜΟΝΩΣΗ (κανένα άγγιγμα στην παραγωγή):
    • ξεχωριστός PersistentClient σε /tmp -> το production chroma.sqlite3 δεν
      ανοίγει ποτέ για γράψιμο.
    • ξεχωριστή διεργασία (docker compose exec) -> τα monkeypatch δεν αγγίζουν τον
      uvicorn server.
    • _save_translation_cache -> no-op.

ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ:
    ΔΙΑΡΡΟΕΣ out_of_corpus = 0 είναι ΑΠΟΚΛΕΙΣΤΙΚΟ — δεν ανταλλάσσεται με σώσεις.
    Ένα prompt περνάει μόνο αν έχει 0 διαρροές ΚΑΙ 0 «χωρίς υλικό».
    Πρόβλεψη πριν το τρέξιμο: οι 5 ooc μένουν σιωπηλές, η v3 ΔΕΝ σώζεται (έχει
    αναφορικό χωρίς υποκείμενο — «ποιο μοντέλο ΠΡΟΤΕΙΝΟΥΝ για τη ΓΡΑΜΜΙΚΗ ΣΧΕΣΗ» —
    ίδια κλάση με τα h012/h016, που κανένας μηχανισμός δεν έλυσε).

ΑΠΟΤΕΛΕΣΜΑ 1ου ΤΡΕΞΙΜΑΤΟΣ (test domains, runs/corrective_domains.csv): B και C
    διαρρέουν 2/5 (o4 Βεζούβιος, o5 στατίνες), ο κανόνας Δ ΔΕΝ τις πιάνει -> ο
    κανόνας Δ ΑΠΟΡΡΙΦΘΗΚΕ. Το probe_oov_rewrite.py έδειξε offline ότι ο κανόνας OOV
    (παρακάτω) τις κόβει ΚΑΙ κρατά τις σώσεις του cloud — αλλά σε rewrites του A.

ΔΕΥΤΕΡΟ ΣΤΑΔΙΟ — CLOUD, ΜΕ ΚΑΝΟΝΑ OOV (25/9/2026):
    Κανόνας OOV: η αναδιατύπωση μένει κομμένη αν έχει ΕΣΤΩ ΕΝΑΝ όρο περιεχομένου
    που δεν εμφανίζεται σε ΚΑΝΕΝΑ in-scope chunk. Ορισμός ΙΔΙΟΣ με το
    probe_oov_rewrite (el_tokenize, >=3 χαρ, όχι αριθμός, εκτός _STOP)· εδώ το
    λεξιλόγιο βγαίνει από τα CHUNKS του ευρετηρίου — αυτό που θα έβλεπε η παραγωγή.
    Το ερώτημα: με prompt ΧΩΡΙΣ πεδίο (B) ή με όρους (C), κρατάει το cloud τις
    σώσεις του A και μένει με 0 διαρροές;

    ΜΕΤΑΦΡΑΣΗ: δεν υπάρχει καταγεγραμμένη για το hard set -> ΖΩΝΤΑΝΗ, με γενικό
    domain (η ΤΕΛΙΚΗ μορφή της παραγωγής χωρίς corpus_descriptor.json). Μόνο 4 από
    τις κομμένες είναι ελληνικές (h005, h012, q020, q049). Μέσα στο ΙΔΙΟ τρέξιμο
    κάθε ερώτηση μεταφράζεται ΜΙΑ φορά και τα A/B/C παίρνουν την ΙΔΙΑ μετάφραση ->
    η σύγκριση των prompts μένει καθαρή.
    CONTROL (ενημερωτικό, --compare-best1): best1 έναντι runs/corr_v1_control.csv.
    Οι ΑΓΓΛΙΚΕΣ ερωτήσεις δεν μεταφράζονται, άρα ΑΝΑΜΕΝΕΤΑΙ ταύτιση· απόκλιση
    εκεί = άλλο ευρετήριο (π.χ. tie-break του RRF στα uuid των chunk ids).

    ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ: ένα prompt + OOV γίνεται υποψήφιο για την
    παραγωγή ΜΟΝΟ αν στο cloud (α) 0 διαρροές, (β) σώσεις ΜΕ υλικό >= όσες του A με
    τον σημερινό κανόνα (= ο σημερινός agent, με ΤΗΝ ΙΔΙΑ μετάφραση), (γ) «χωρίς
    υλικό» <= του A. Αν
    περάσουν και τα δύο, προτιμάται το B: καμία εξάρτηση από metadata, καμία
    επιφάνεια λεξιλογίου corpus μέσα στο prompt.
    Πρόβλεψη πριν το τρέξιμο: το B χάνει >= 1 σώση του A (το «Cloud resource
    pricing…» του h008 έχει «cloud» επειδή το έβαλε το prompt)· 0 διαρροές με OOV.

ΚΟΣΤΟΣ: μία σύντομη κλήση Gemini ανά (κομμένη ερώτηση × prompt × επανάληψη).
Test domains: ~6 κομμένες × 3 = ~18 κλήσεις. Cloud: ~11 × 3 = ~33 + ~25 ζωντανές
μεταφράσεις (οι ελληνικές ερωτήσεις ΚΑΙ των δύο σετ: το 1ο πέρασμα τρέχει σε όλες
για να φανεί ποιες κόβει το gate — παράπλευρα μετράει και αν η γενική μετάφραση
κόβει κάποια in-corpus του κύριου σετ).

    docker compose exec backend python evaluation/probe_corrective_domains.py \
        --papers evaluation/test_papers/cureus-0015-00000046486.pdf \
                 evaluation/test_papers/s41598-017-03833-3.pdf \
        --golden evaluation/golden_test_domains.jsonl \
        --translations evaluation/runs/domain_glossary_JK.csv \
        --csv evaluation/runs/corrective_domains.csv

    docker compose exec backend python evaluation/probe_corrective_domains.py \
        --papers evaluation/test_papers/cloud/*.pdf \
        --golden evaluation/golden_hard_paraphrase.jsonl evaluation/golden_set_50.jsonl \
        --compare-best1 evaluation/runs/corr_v1_control.csv \
        --csv evaluation/runs/corrective_cloud.csv
"""
import argparse
import asyncio
import csv
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Το import του ai_core φορτώνει bge-m3 + reranker: 40-60 s ΧΩΡΙΣ έξοδο. Χωρίς αυτό
# το μήνυμα μοιάζει κολλημένο -> Ctrl+C (συνέβη στο πρώτο τρέξιμο).
print("Φόρτωση μοντέλων (40-60 s χωρίς έξοδο) — ΜΗΝ το διακόψεις...", flush=True)

import chromadb  # noqa: E402

import ai_core  # noqa: E402
import corpus_glossary  # noqa: E402
import gemini_rest  # noqa: E402

TEST_USER = 999_998          # εικονικός ιδιοκτήτης των test chunks (authz)
TEST_DB = "/tmp/probe_corrective_domains_chroma"  # noqa: S108  εφήμερο, μέσα στο container

# Κοινή ουρά και για τα τρία — ΑΥΤΟΥΣΙΑ από το _CORRECTIVE_PROMPT της παραγωγής.
_TAIL = (
    "Output ONLY the rewritten query. No quotes, no extra text.\n\n"
    "Original query: {query}\nRewritten query:"
)
_RULES = (
    "Replace vague or conversational wording with the "
    "terms a paper would actually use. Do NOT add facts, system names or numbers "
    "that are not implied by the original question."
)

PROMPT_B_GENERIC = (
    "The following search query returned no sufficiently relevant results from a "
    "corpus of research documents.\n\n"
    "Rewrite it as a keyword-rich search query using the standard technical "
    "terminology of the field that the question itself is about. " + _RULES + "\n\n"
    + _TAIL
)

# {terms} συμπληρώνεται ΠΡΙΝ ανατεθεί στο ai_core (μένει μόνο το {query}).
PROMPT_C_TERMS = (
    "The following search query returned no sufficiently relevant results from a "
    "corpus of research documents whose key terms include: {terms}.\n\n"
    "Rewrite it as a keyword-rich search query using the standard technical "
    "terminology of that field. " + _RULES + " If the question is about a "
    "topic OUTSIDE that field, return it unchanged.\n\n"
    + _TAIL
)


def load_golden(paths: list[str]) -> list[dict]:
    tests = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            tests.extend(json.loads(line) for line in f if line.strip())
    return tests


def content_terms(text: str) -> list[str]:
    """ΙΔΙΟΣ ορισμός με probe_oov_rewrite.content_terms (προεγγεγραμμένος 25/9)."""
    return [t for t in dict.fromkeys(ai_core.el_tokenize(text))
            if len(t) >= 3 and not t.isdigit() and t not in corpus_glossary._STOP]


def coverage(keywords: list[str], texts: list[str]) -> float:
    """ΙΔΙΟΣ ορισμός με eval_engine.evaluate_retrieval / probe_domain_glossary."""
    if not texts or not keywords:
        return 0.0
    joined = "\n".join(texts).lower()
    return sum(1 for k in keywords if k.lower() in joined) / len(keywords) * 100.0


def load_translations(path: str, condition: str) -> dict:
    """id -> (μετάφραση, best_logit, coverage) της καταγεγραμμένης συνθήκης."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["condition"] == condition:
                out[r["id"]] = (r["translation"], float(r["best_logit"]),
                                r["coverage"])
    if not out:
        raise SystemExit(f"Καμία γραμμή για τη συνθήκη {condition} στο {path}")
    return out


# --- Spies: ΜΟΝΟ διαβάζουν, επιστρέφουν αυτούσια ------------------------------
_last = {"best": None, "rewrite": None}


def _install_spies():
    orig_predict = ai_core.reranker.predict

    def predict_spy(pairs, **kw):
        scores = orig_predict(pairs, **kw)
        if len(scores):
            _last["best"] = float(max(scores))
        return scores

    ai_core.reranker.predict = predict_spy

    orig_gen = gemini_rest.generate_once

    async def gen_spy(prompt, **kw):
        text = await orig_gen(prompt, **kw)
        _last["rewrite"] = text.strip(" \"'\n")
        return text

    # ai_core καλεί `gemini_rest.generate_once` μέσω του module -> αρκεί εδώ.
    gemini_rest.generate_once = gen_spy


def verdict(ooc: bool, passes: bool, cov2: float) -> str:
    if ooc:
        return "ΔΙΑΡΡΟΗ" if passes else "σιωπή"
    if not passes:
        return "κομμένο"
    return "ΣΩΘΗΚΕ" if cov2 > 0 else "ΧΩΡΙΣ ΥΛΙΚΟ"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", required=True)
    ap.add_argument("--golden", nargs="+", required=True)
    ap.add_argument("--translations", default=None,
                    help="CSV με καταγεγραμμένες μεταφράσεις (probe_domain_glossary). "
                         "Χωρίς αυτό: ΖΩΝΤΑΝΗ μετάφραση με γενικό domain — ΜΗ επαναλήψιμο.")
    ap.add_argument("--condition", default="CD_constrained")
    ap.add_argument("--variants", default="A_cloud,B_generic,C_terms")
    ap.add_argument("--repeats", type=int, default=1,
                    help="αναδιατυπώσεις ανά (ερώτηση, prompt). temperature 0.1 -> "
                         "7/9 ταυτόσημες στο cloud (probe_corrective_attempts)")
    ap.add_argument("--margin", type=float, default=0.43,
                    help="ελάχιστο Δ = best2 − best1 (προεγγεγραμμένο: μέσο του "
                         "καθαρού διαστήματος (−0.02, +0.88) του cloud)")
    ap.add_argument("--compare-best1", default=None,
                    help="CSV με στήλες id,best1 (π.χ. runs/corr_v1_control.csv): "
                         "ΕΝΗΜΕΡΩΤΙΚΗ σύγκριση του 1ου περάσματος — δεν σταματά")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    ai_core._save_translation_cache = lambda: None
    corr_floor = ai_core.CORRECTIVE_MIN_SCORE   # το σημερινό −3.8, πριν το −inf
    gate = ai_core.MIN_RERANK_SCORE

    # --- Απομονωμένο ευρετήριο ---
    if os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB)
    client = chromadb.PersistentClient(path=TEST_DB)
    col = client.get_or_create_collection(
        name="probe_corrective_domains",
        embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    # Το `cloud/*.pdf` φτάνει ΚΥΡΙΟΛΕΚΤΙΚΟ: το PowerShell δεν αναπτύσσει globs σε
    # native εντολές και το docker exec δεν περνά από shell. Το αναπτύσσουμε εδώ.
    papers = [p for pat in args.papers for p in (sorted(glob.glob(pat)) or [pat])]
    missing = [p for p in papers if not os.path.exists(p)]
    if missing:
        print(f"!! Δεν βρέθηκαν: {missing} — σταματάω")
        return 1
    print(f"Ingest {len(papers)} papers στο {TEST_DB} (embeddings σε CPU, "
          "~1 λεπτό ανά 100 chunks) ...", flush=True)
    for i, path in enumerate(papers):
        ok = ai_core.ingest_pdf(path, os.path.basename(path), user_id=TEST_USER,
                                is_public=False, doc_id=i)
        print(f"  {os.path.basename(path)}: {'OK' if ok else 'ΚΕΝΟ'}", flush=True)
    ai_core._bump_corpus_version()
    _install_spies()

    tests = load_golden(args.golden)
    fixed = None
    if args.translations:
        fixed = load_translations(args.translations, args.condition)
        for t in tests:
            if t["id"] in fixed:
                ai_core._translation_cache[t["question"]] = fixed[t["id"]][0]
        print(f"Μεταφράσεις ΣΤΑΘΕΡΕΣ από {args.translations} [{args.condition}]")
    else:
        # Η τελική μορφή της παραγωγής: γενικό domain, όροι ανά αρχείο.
        ai_core._CORPUS_DOMAIN, ai_core._CORPUS_TERMS = ai_core._FALLBACK_DOMAIN, ""
        ai_core._translation_cache.clear()
        print("!! ΖΩΝΤΑΝΗ μετάφραση (γενικό domain) — το 1ο πέρασμα ΔΕΝ είναι επαναλήψιμο")

    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    allowed_ids = col.get(where=ai_core._build_where(None, TEST_USER), include=[])["ids"]
    scope_domain, scope_terms = corpus_glossary.assemble(allowed_ids, idx)
    print(f"Scope: {len(allowed_ids)} chunks · όροι: {scope_terms[:120]}...")
    # Λεξιλόγιο του κανόνα OOV: τα in-scope CHUNKS του cached BM25 index, με τον
    # ΙΔΙΟ tokenizer — ακριβώς ό,τι θα είχε στη διάθεσή του ο κανόνας στην παραγωγή.
    vocab: set[str] = set()
    for cid in allowed_ids:
        vocab.update(ai_core.el_tokenize(idx["texts"][idx["pos"][cid]]))
    print(f"Λεξιλόγιο OOV: {len(vocab)} διακριτά tokens")

    # --- 1ο πέρασμα: ο ΠΡΑΓΜΑΤΙΚΟΣ search_documents, με τον agent ΚΛΕΙΣΤΟ ---
    ai_core.ENABLE_CORRECTIVE = False
    print(f"\n===== 1ο ΠΕΡΑΣΜΑ (gate {gate}) =====")
    first, control_bad = {}, []
    for t in tests:
        tr = await ai_core.optimize_query(t["question"], domain=scope_domain,
                                          terms=scope_terms)
        _last["best"] = None
        pages = await ai_core.search_documents(t["question"], target_filenames=None,
                                               user_id=TEST_USER)
        best1 = _last["best"]
        if best1 is None:
            print(f"!! {t['id']}: δεν έτρεξε reranker (κενό scope;) -> σταματάω")
            return 1
        ooc = t.get("category") == "out_of_corpus"
        cov1 = None if ooc else coverage(t["keywords"], [p for p, _m in pages])
        first[t["id"]] = (tr, best1, not pages, cov1)
        ctrl = ""
        if fixed and t["id"] in fixed:
            exp = fixed[t["id"]][1]
            if abs(best1 - exp) > 0.01:
                control_bad.append(t["id"])
                ctrl = f"  !! CONTROL: αναμενόταν {exp:+.2f}"
        cut = "ΚΟΜΜΕΝΟ" if not pages else "περνάει"
        cov_s = "   ooc" if ooc else f"{cov1:5.1f}%"
        print(f"  {t['id']:<4} {cut:<8} best1 {best1:+6.2f}  cov {cov_s}  -> {tr[:50]}{ctrl}")

    if control_bad:
        print(f"\n!! CONTROL ΑΠΕΤΥΧΕ σε {control_bad}: το ευρετήριο ΔΕΝ αναπαράγει τη "
              "μέτρηση -> τα αποτελέσματα ΔΕΝ διαβάζονται. Σταματάω.")
        return 1
    if fixed:
        print("\nCONTROL OK: κάθε best1 αναπαράγει το CSV (±0.01).")

    if args.compare_best1:
        # ΕΝΗΜΕΡΩΤΙΚΟ: στις ΑΓΓΛΙΚΕΣ (καμία μετάφραση) αναμένεται ταύτιση· στις
        # ελληνικές η ζωντανή μετάφραση μπορεί να δώσει άλλο ερώτημα.
        with open(args.compare_best1, encoding="utf-8") as f:
            old = {r["id"]: float(r["best1"]) for r in csv.DictReader(f) if r.get("best1")}
        qtext = {t["id"]: t["question"] for t in tests}
        print(f"\n----- σύγκριση best1 με {os.path.basename(args.compare_best1)} -----")
        for qid, b_old in old.items():
            if qid not in first:
                continue
            lang = "GR" if ai_core._has_greek(qtext[qid]) else "EN"
            diff = first[qid][1] - b_old
            flag = "  !! ΑΠΟΚΛΙΣΗ (EN: άλλο ευρετήριο;)" if lang == "EN" and abs(diff) > 0.01 else ""
            print(f"  {qid:<5} {lang}  τότε {b_old:+6.2f}  τώρα {first[qid][1]:+6.2f}  "
                  f"Δ {diff:+5.2f}{flag}")

    cut_tests = [t for t in tests if first[t["id"]][2]]
    print(f"\nΚομμένες από το gate: {[t['id'] for t in cut_tests]} "
          f"-> ο agent τρέχει ΜΟΝΟ σε αυτές")

    # --- 2ο πέρασμα: η ΠΡΑΓΜΑΤΙΚΗ _corrective_retry, κατώφλι offline ---
    prompts = {
        "A_cloud": ai_core._CORRECTIVE_PROMPT,
        "B_generic": PROMPT_B_GENERIC,
        "C_terms": PROMPT_C_TERMS.replace("{terms}", scope_terms),
    }
    ai_core.ENABLE_CORRECTIVE = True
    ai_core.CORRECTIVE_MIN_SCORE = float("-inf")
    rows = []
    for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
        ai_core._CORRECTIVE_PROMPT = prompts[vname]
        print(f"\n===== {vname} =====")
        for t in cut_tests:
            tr, best1, _cut, _cov1 = first[t["id"]]
            ooc = t.get("category") == "out_of_corpus"
            for rep in range(args.repeats):
                _last["rewrite"] = None
                sf = await ai_core._corrective_retry(tr, best1, allowed_ids, idx, dm)
                rw = _last["rewrite"] or ""
                if sf is None:
                    skip = ("identical" if rw and rw.strip().lower() == tr.strip().lower()
                            else "error")
                    best2, delta, cov2, n_pages = None, None, 0.0, 0
                else:
                    skip = ""
                    best2 = float(sf[0][0])
                    delta = best2 - best1
                    pages = ai_core._expand_to_pages(sf[:ai_core.EXPAND_INPUT],
                                                     ai_core.MAX_PAGES, TEST_USER)
                    n_pages = len(pages)
                    cov2 = 0.0 if ooc else coverage(t["keywords"], [p for p, _m in pages])
                pass_today = best2 is not None and best2 >= corr_floor
                pass_delta = pass_today and delta >= args.margin
                absent = [w for w in content_terms(rw) if w not in vocab] if sf else []
                pass_oov = pass_today and not absent
                row = {
                    "variant": vname, "id": t["id"], "category": t.get("category"),
                    "rep": rep, "translation": tr, "rewrite": rw,
                    "best1": round(best1, 3),
                    "best2": "" if best2 is None else round(best2, 3),
                    "delta": "" if delta is None else round(delta, 3),
                    "cov2": "" if ooc else round(cov2, 1), "pages2": n_pages,
                    "skip": skip,
                    "today": verdict(ooc, pass_today, cov2),
                    "with_delta": verdict(ooc, pass_delta, cov2),
                    "absent": ",".join(absent),
                    "with_oov": verdict(ooc, pass_oov, cov2),
                }
                rows.append(row)
                b2 = "   n/a" if best2 is None else f"{best2:+6.2f}"
                d = "   n/a" if delta is None else f"{delta:+6.2f}"
                c = "   ooc" if ooc else f"{cov2:5.1f}%"
                print(f"  {t['id']:<4} b1 {best1:+6.2f}  b2 {b2}  Δ {d}  cov2 {c}  "
                      f"σήμερα={row['today']:<11} μεOOV={row['with_oov']:<11} "
                      f"{skip}\n        -> {rw[:90]}"
                      + (f"\n        df=0: {row['absent']}" if absent else ""))

    # --- Σύνοψη ---
    n_ooc = sum(1 for t in cut_tests if t.get("category") == "out_of_corpus")
    print("\n" + "#" * 78)
    print(f"ΣΥΝΟΨΗ  (κατώφλι 2ου περάσματος {corr_floor} · margin Δ ≥ {args.margin:+.2f})")
    print(f"{'prompt':<10} {'κανόνας':<14} {'σώθηκαν':>8} {'χωρίς υλικό':>12} "
          f"{'διαρροές':>9}")
    for vname in dict.fromkeys(r["variant"] for r in rows):
        vr = [r for r in rows if r["variant"] == vname]
        for rule, key in (("σήμερα", "today"), ("+ Δ", "with_delta"), ("+ OOV", "with_oov")):
            saved = sum(r[key] == "ΣΩΘΗΚΕ" for r in vr)
            nomat = sum(r[key] == "ΧΩΡΙΣ ΥΛΙΚΟ" for r in vr)
            leaks = sorted({r["id"] for r in vr if r[key] == "ΔΙΑΡΡΟΗ"})
            flag = f"  <-- {', '.join(leaks)}" if leaks else ""
            print(f"{vname:<10} {rule:<14} {saved:>8} {nomat:>12} "
                  f"{len(leaks):>5}/{n_ooc}{flag}")
        d_ooc = [r["delta"] for r in vr if r["category"] == "out_of_corpus" and r["delta"] != ""]
        d_in = [r["delta"] for r in vr if r["category"] != "out_of_corpus"
                and r["delta"] != "" and r["cov2"] not in ("", 0.0)]
        if d_ooc or d_in:
            print(f"{'':<10} Δ ooc max {max(d_ooc) if d_ooc else 'n/a'} · "
                  f"Δ με υλικό min {min(d_in) if d_in else 'n/a'}")
    print("\nΟι διαρροές είναι ΚΡΙΤΗΡΙΟ ΑΠΟΚΛΕΙΣΜΟΥ — δεν ανταλλάσσονται με σώσεις.")

    # --- Κριτήριο γραμμένο πριν (βλ. docstring): B/C + OOV έναντι A σήμερα ---
    def count(vname, key, label):
        return sum(r[key] == label for r in rows if r["variant"] == vname)

    if any(r["variant"] == "A_cloud" for r in rows):
        a_saved = count("A_cloud", "today", "ΣΩΘΗΚΕ")
        a_nomat = count("A_cloud", "today", "ΧΩΡΙΣ ΥΛΙΚΟ")
        print(f"\nΚΡΙΤΗΡΙΟ (γραμμένο πριν) — βάση: A σήμερα = {a_saved} σώσεις, "
              f"{a_nomat} χωρίς υλικό")
        for vname in ("B_generic", "C_terms"):
            if not any(r["variant"] == vname for r in rows):
                continue
            leaks = count(vname, "with_oov", "ΔΙΑΡΡΟΗ")
            saved = count(vname, "with_oov", "ΣΩΘΗΚΕ")
            nomat = count(vname, "with_oov", "ΧΩΡΙΣ ΥΛΙΚΟ")
            ok = leaks == 0 and saved >= a_saved and nomat <= a_nomat
            print(f"  {vname} + OOV: διαρροές {leaks} · σώσεις {saved} · χωρίς υλικό {nomat}"
                  f"  -> {'ΥΠΟΨΗΦΙΟ' if ok else 'ΑΠΟΡΡΙΠΤΕΤΑΙ'}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nΓράφτηκε: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
