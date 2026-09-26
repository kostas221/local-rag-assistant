"""Corpus glossary ΑΝΑ ΕΓΓΡΑΦΟ — τροφοδοτεί το translate-then-retrieve (optimize_query).

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (η σχεδίαση που αντικαθιστά):
    Ο παλιός descriptor ήταν ΚΑΘΟΛΙΚΟΣ, φτιαγμένος ΜΙΑ φορά από CLI
    (build_corpus_descriptor.py -> corpus_descriptor.json). Απορρίφθηκε ως σχεδίαση:
    (α) ο χρήστης δεν τρέχει CLI σε κάθε upload· (β) για να ξαναχτιστεί σωστά όταν
    προστεθεί νέο αρχείο ΙΔΙΟΥ θέματος θα έπρεπε να ΣΒΗΣΤΟΥΝ τα υπάρχοντα. Εδώ η
    ορολογία εξάγεται ΑΝΑ ΕΓΓΡΑΦΟ στο ingest και γράφεται στα ΙΔΙΑ metadata που το
    query path ήδη φορτώνει (terms/domain δίπλα στο doc_id/file_name) — καμία δεύτερη
    πηγή αλήθειας, κανένα CLI, κανένα σβήσιμο.

Ο ΛΟΓΟΣ ΠΟΥ ΕΙΝΑΙ ΞΕΧΩΡΙΣΤΟ (leaf) MODULE:
    Το ai_core κάνει `import corpus_glossary`. Άρα ΕΔΩ ΔΕΝ επιτρέπεται `import ai_core`
    (κυκλικό). Το μόνο βαρύ που χρειάζεται ο opt-in LLM δρόμος είναι το gemini_rest,
    που είναι κι αυτό leaf (httpx μόνο) και εισάγεται LAZY, ώστε ο default στατιστικός
    δρόμος να μην έχει ΚΑΜΙΑ εξάρτηση δικτύου/μοντέλου.

ΠΟΙΑ ΜΕΘΟΔΟ ΔΙΑΛΕΞΑΜΕ — ΚΑΙ ΓΙΑΤΙ ΣΤΑΤΙΣΤΙΚΗ, ΟΧΙ LLM (probe_domain_glossary.py):
    Μετρήθηκαν και οι δύο σε ΔΥΟ σετ (κύριο cloud n=45· test-domains cardiology+
    volcanology n=20 + 5 out_of_corpus). Το ΚΡΙΣΙΜΟ αποτέλεσμα:
        cleaned-statistical (CD)   coverage 95.0%   ooc 5/5 σιωπηλά
        LLM specific    (CG)        coverage 95.0%   ooc 5/5 σιωπηλά
    ΙΣΟΠΑΛΙΑ και στις δύο μετρικές, στα δύο σετ. Η στατιστική είναι ΔΩΡΕΑΝ,
    ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΗ και ΧΩΡΙΣ εξάρτηση δικτύου στο upload· ο LLM βάζει μία κλήση
    Gemini + μη-ντετερμινισμό (±1 keyword ~ ±0.74pp coverage) + latency στο ingest.
    Ίδιο μοτίβο απόρριψης με enrichment/decomposition/disambiguation: μηδέν μετρημένο
    κέρδος -> δεν πληρώνεται το κόστος. Ο LLM μένει ως ΤΕΚΜΗΡΙΩΜΕΝΟ opt-in (use_llm=True)
    με fallback στη στατιστική — ένας διακόπτης, αν κάποιο μελλοντικό πραγματικό
    upload δείξει πεδίο όπου η στατιστική διαλέγει γενικούς-αλλά-συχνούς όρους.

ΤΙ ΕΜΠΟΔΙΖΕΙ ΤΙΣ ΔΙΑΡΡΟΕΣ (και ΔΕΝ είναι το είδος των όρων):
    Το glossary σπρώχνει το ερώτημα προς το λεξιλόγιο του corpus — ΑΚΡΙΒΩΣ ο μηχανισμός
    που διέρρευσε 4 προηγούμενες ιδέες. Στο test-domains, η «prefer the vocabulary»
    οδηγία έκανε το «Πότε εξερράγη ο Βεζούβιος;» -> "Vesuvius eruption RATE" (+0.49,
    ΔΙΑΡΡΟΗ) βάζοντας corpus-λέξη που ΔΕΝ υπάρχει στην ερώτηση. Η ΠΕΡΙΟΡΙΣΤΙΚΗ οδηγία
    του optimize_query («όρο glossary ΜΟΝΟ ως μετάφραση λέξης που υπάρχει· ΠΟΤΕ προσθήκη·
    ΠΟΤΕ σε κύριο όνομα») το έκλεισε: ooc 4/5 -> 5/5, coverage αμετάβλητο. Δηλαδή η
    άμυνα είναι στο ai_core.optimize_query, ΟΧΙ εδώ — αυτό το module απλώς παρέχει τους
    όρους. Γι' αυτό το assemble ΔΕΝ βάζει καπάκι στο πλήθος όρων: το πλήθος δεν είναι
    ο μοχλός της διαρροής (μετρημένο), η οδηγία είναι.
"""
import re
import unicodedata
from collections import Counter

# --- Tokenizer: ΙΔΙΟΣ με το probe (suggest_keywords._tokens) που παρήγαγε τα
# μετρημένα 95% -- ώστε η παραγωγή να αναπαράγει ό,τι μετρήθηκε, όχι κάτι κοντινό.
_TERM_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{2,}")

# Το STOP του suggest_keywords, ΕΝΣΩΜΑΤΩΜΕΝΟ αυτούσιο (το leaf module δεν εισάγει
# κώδικα του evaluation/). Πλουσιότερο από ένα σκέτο stopword set: κόβει και
# «figure/table/section/paper/fig/et/al» -- boilerplate ακαδημαϊκού κειμένου που
# αλλιώς θα έμπαινε ισότιμα με πραγματική ορολογία στον μεταφραστή. Το `.split()`
# σε multiline string κρατά τα tokens ΑΣΦΑΛΩΣ (τα newlines είναι whitespace)· η
# list-literal διόρθωση του SIM905 θα έβγαζε 130 όρους σε μία δυσανάγνωστη γραμμή,
# με ρίσκο να χαθεί όρος -> αλλάζει το ΜΕΤΡΗΜΕΝΟ STOP set. Γι' αυτό noqa.
_STOP = frozenset(
    """
the a an and or but if then than that this these those of in on at to for from
with without by as is are was were be been being it its it's they them their
we our you your he she his her not no nor so such can could may might must
will would shall should do does did done have has had having when where which
who whom whose what why how all any both each few more most other some only own
same too very s t just don now also into over under between during about above
below up down out off again further once here there both while because until
one two three first second new use used using make makes made way ways well
however thus therefore e.g i.e et al fig figure table section paper work
""".split()  # noqa: SIM905
)


def _tokens(text: str) -> list[str]:
    """Πεζά tokens >=3 χαρακτήρων, χωρίς stopwords. Ίδιο με το probe."""
    return [w for w in _TERM_RE.findall(text.lower()) if w not in _STOP]


def document_terms(page_texts: list[str], top: int = 15,
                   boilerplate_ratio: float = 0.9) -> list[str]:
    """Έως `top` χαρακτηριστικοί όροι του εγγράφου — η μετρημένη μέθοδος CC/CD.

    Αλγόριθμος: tf πάνω σε όλα τα tokens, df πάνω στις ΣΕΛΙΔΕΣ. Κρατάμε όρους με
    df>=2 (όχι hapax: μία μόνο εμφάνιση = πιθανό τυπογραφικό/όνομα), ταξινομημένους
    κατά συχνότητα (θέλουμε το ΧΑΡΑΚΤΗΡΙΣΤΙΚΟ λεξιλόγιο, όχι διακριτικούς όρους).
    Δύο καθαρισμοί που μετρήθηκε ότι έβγαζαν ~1/3 θόρυβο από το raw:
      1. BOILERPLATE: όρος σε >=90% των σελίδων = header/footer (π.χ. όνομα
         περιοδικού «cureus»). Ενεργό ΜΟΝΟ με >=6 σελίδες — σε 3σέλιδο έγγραφο
         «σε κάθε σελίδα» σημαίνει θέμα, όχι footer.
      2. ΕΝΙΚΟΣ/ΠΛΗΘΥΝΤΙΚΟΣ: «patient»+«patients» έπιαναν δύο θέσεις για την ίδια
         λέξη. Κρατάμε ΜΟΝΟ την πρώτη (= συχνότερη, αφού είναι ταξινομημένο).
    Ο καθαρισμός ορίζεται ΔΟΜΙΚΑ (συχνότητα/κατάληξη), όχι σημασιολογικά -> δεν
    χρειάζεται να μαντέψουμε πεδίο, δουλεύει σε ΟΠΟΙΟΔΗΠΟΤΕ θέμα ανεβάσει ο χρήστης.
    """
    n_pages = len(page_texts)
    tf, df = Counter(), Counter()
    for txt in page_texts:
        toks = _tokens(txt)
        tf.update(toks)
        df.update(set(toks))

    boiler: set[str] = set()
    if n_pages >= 6:
        boiler = {w for w, d in df.items() if d >= boilerplate_ratio * n_pages}

    cand = [(c, w) for w, c in tf.items() if df[w] >= 2 and w not in boiler]
    cand.sort(key=lambda x: (-x[0], x[1]))

    out, seen_roots = [], set()
    for _c, w in cand:
        root = w[:-3] + "y" if w.endswith("ies") else w.rstrip("s")
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(w)
        if len(out) >= top:
            break
    return out


def statistical_glossary(page_texts: list[str], top: int = 15) -> dict:
    """Ο DEFAULT (δωρεάν, ντετερμινιστικός). domain κενό επίτηδες: η στατιστική
    δεν ονομάζει πεδίο -> το optimize_query πέφτει στο καθολικό _CORPUS_DOMAIN,
    που μετρήθηκε ουδέτερο (το domain string δεν κουνάει coverage ούτε διαρροές)."""
    return {"domain": "", "terms": ", ".join(document_terms(page_texts, top))}


# --- Ίδιο prompt με τον build_corpus_descriptor.py / το probe (CB/CG) ----------
_LLM_PROMPT = (
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


async def llm_glossary(page_texts: list[str], *, model: str, api_key: str,
                       thinking_budget: int = 1024) -> dict:
    """OPT-IN: domain + terms από το Gemini, ανά έγγραφο. Δείγμα ~15 αποσπασμάτων
    (600 χαρ το καθένα) απλωμένο σε όλο το έγγραφο. gemini_rest = LAZY import ώστε
    ο στατιστικός δρόμος να μη σέρνει καμία εξάρτηση."""
    import gemini_rest  # lazy import: μόνο ο opt-in δρόμος το χρειάζεται

    step = max(1, len(page_texts) // 15)
    sample = "\n---\n".join(t[:600] for t in page_texts[::step][:15])
    raw = (await gemini_rest.generate_once(
        _LLM_PROMPT.format(sample=sample), model=model, api_key=api_key,
        thinking_budget=thinking_budget, max_output_tokens=2048)).strip()
    domain, terms = "", ""
    for line in raw.splitlines():
        if line.upper().startswith("DOMAIN:"):
            domain = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TERMS:"):
            terms = line.split(":", 1)[1].strip()
    # Κενό αποτέλεσμα (κακό format/άδεια απάντηση) -> σήμα στον caller να κάνει fallback.
    if not terms:
        raise ValueError("LLM glossary: κενοί όροι στην απάντηση")
    return {"domain": domain, "terms": terms}


def extract_glossary(page_texts: list[str], *, use_llm: bool = False,
                     model: str | None = None, api_key: str | None = None) -> dict:
    """Ο ΕΝΙΑΙΟΣ dispatcher που καλεί το ingest. Επιστρέφει {"domain","terms"}.

    Default (use_llm=False): στατιστικό, sync, μηδέν δίκτυο — ο μετρημένος νικητής.
    use_llm=True: δοκιμάζει τον LLM· σε ΟΠΟΙΑΔΗΠΟΤΕ αποτυχία (δίκτυο, quota, κακό
    format, ή τρέχον event loop που εμποδίζει το asyncio.run) πέφτει ΣΙΩΠΗΛΑ στη
    στατιστική. Δηλαδή ο opt-in δεν μπορεί ΠΟΤΕ να αφήσει έγγραφο χωρίς glossary."""
    stat = statistical_glossary(page_texts)
    if not use_llm:
        return stat
    if not (model and api_key):
        return stat
    try:
        import asyncio  # lazy: μόνο ο opt-in δρόμος φτάνει εδώ
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # κανένα loop -> ασφαλές asyncio.run (το ingest τρέχει σε sync worker)
        else:
            return stat  # υπάρχει loop -> δεν μπορούμε να κάνουμε nested run
        g = asyncio.run(llm_glossary(page_texts, model=model, api_key=api_key))
        return g
    except Exception:
        return stat


def assemble(allowed_ids: list[str], idx: dict) -> tuple[str, str]:
    """QUERY-TIME: ένωσε domain+terms ΜΟΝΟ από τα αρχεία που είναι στο scope.

    Παίρνει τα ήδη φιλτραρισμένα-κατά-authz `allowed_ids` και το cached BM25 index
    (idx["pos"] = {id: θέση}, idx["metas"] ευθυγραμμισμένο με idx["ids"]). Χαρτογραφεί
    κάθε chunk-id στο metadata του με ΜΗΔΕΝ επιπλέον κλήση στη βάση, και ενώνει ανά
    ΑΡΧΕΙΟ (dedup κατά file_name: όλα τα chunks ενός αρχείου έχουν ΙΔΙΟ terms/domain,
    δεν θέλουμε να μετρήσει το αρχείο 40 φορές).

    Η ένωση είναι ΤΑΥΤΟΣΗΜΗ με αυτήν που μετρήθηκε στο probe (dict.fromkeys ->
    order-preserving, χωρίς διπλά): domains με «; », terms με «, ». Επιστρέφει
    (domain, terms) — κενά αν κανένα in-scope αρχείο δεν έχει glossary metadata
    (π.χ. legacy chunks πριν το backfill), οπότε το optimize_query πέφτει στα καθολικά.
    """
    pos = idx.get("pos") or {}
    metas = idx.get("metas") or []
    domains: list[str] = []
    term_items: list[str] = []
    seen_files: set = set()
    for cid in allowed_ids:
        p = pos.get(cid)
        if p is None or p >= len(metas):
            continue
        m = metas[p] or {}
        fn = m.get("file_name")
        if fn in seen_files:
            continue
        seen_files.add(fn)
        d = (m.get("domain") or "").strip()
        if d:
            domains.append(d)
        t = (m.get("terms") or "").strip()
        if t:
            term_items.extend(x.strip() for x in t.split(",") if x.strip())
    domain = "; ".join(dict.fromkeys(domains))
    terms = ", ".join(dict.fromkeys(term_items))
    return domain, terms


def _normalize(text: str) -> str:
    """NFKC (ΟΧΙ NFKD -> θα έσπαγε ελληνικούς τόνους). Δεν χρησιμοποιείται στο
    hot path· εδώ για συνέπεια αν κληθεί document_terms σε μη κανονικοποιημένο
    κείμενο εκτός ingest (backfill/probe)."""
    return unicodedata.normalize("NFKC", text)
