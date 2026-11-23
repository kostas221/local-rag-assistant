"""Στήριξη ανά ισχυρισμό (βήμα 5, docs/CLAIM_SUPPORT_PREREG_2026-11-22.md).

ΤΙ ΚΑΝΕΙ: παίρνει μια ΤΕΛΕΙΩΜΕΝΗ απάντηση και τις σελίδες που δόθηκαν στο μοντέλο, και για κάθε
πρόταση με ισχυρισμό απαντά σε τρία χωριστά ερωτήματα:
    (α) υπάρχει παραπομπή [S#];
    (β) ανοίγει το σωστό σημείο: το χωρίο που τη στηρίζει βρίσκεται ΑΥΤΟΥΣΙΟ σε σελίδα που
        παραπέμπεται;
    (γ) το χωρίο στηρίζει ΟΛΗ την πρόταση (αριθμούς, μονάδες, έτη, πεδίο, αιτιώδεις σχέσεις);
Το (β) και ο έλεγχος αριθμών είναι ντετερμινιστικά. Το (γ) το κρίνει ΜΙΑ κλήση μοντέλου ανά
απάντηση, και ο κώδικας ελέγχει ότι το χωρίο που επέστρεψε υπάρχει πράγματι στη σελίδα.

ΤΙ ΔΕΝ ΚΑΝΕΙ: δεν αλλάζει την απάντηση και δεν αγγίζει το prompt της γέννησης. Άρα οι σωστές
αρνήσεις μένουν όπως είναι· το μόνο που μπορεί να χαλάσει είναι να σημαδευτεί λάθος μια σωστή
πρόταση (μετριέται στους μάρτυρες ελέγχου).

ΠΡΟΕΛΕΥΣΗ: ο έλεγχος αυτούσιου χωρίου και οι κανόνες αριθμών είναι από το `evidence_cards.py` του
κλάδου `library-frozen` (16/11/2026), προσαρμοσμένοι σε σελίδες άρθρων αντί για κάρτες.

Leaf module: μόνο stdlib, κανένα import από το project (δοκιμάζεται χωρίς Docker).
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field

MIN_CLAIM_WORDS = 4
MAX_QUOTE_WORDS = 40

# ---------------------------------------------------------------------------
# Κανονικοποίηση και αυτούσιο χωρίο
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_QUOTES = {ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'}


def normalize_for_match(text: str) -> str:
    """NFKC, ενιαίες παύλες και εισαγωγικά, ενωμένες λέξεις που έσπασαν με παύλα στο τέλος
    γραμμής, πεζά, ενιαία κενά. Ίδια μορφή για το χωρίο και για τη σελίδα."""
    t = unicodedata.normalize("NFKC", text or "").translate(_DASHES).translate(_QUOTES).lower()
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


def _alnum(text: str) -> str:
    """Μόνο γράμματα και ψηφία, πεζά, χωρίς τόνους: η ΣΕΙΡΑ των γραμμάτων μένει ίδια, φεύγουν
    κενά, παύλες, εισαγωγικά, αλλαγές γραμμής."""
    t = unicodedata.normalize("NFKD", text or "").lower()
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", t)


MIN_QUOTE_ALNUM = 15                       # «0.5» ή «the cloud» δεν δείχνουν σημείο
# Πρώτα κομμάτια χωρισμένα με κενή γραμμή ή «...»· μόνο αν δεν βρεθούν, γραμμή-γραμμή (κελιά).
_FRAGMENT_SPLITS = (re.compile(r"\.\.\.|…|\n\s*\n"), re.compile(r"\.\.\.|…|\n"))


def quote_in_source(quote: str, source_text: str, min_len: int = MIN_QUOTE_ALNUM) -> bool:
    """Το χωρίο υπάρχει αυτούσιο στη σελίδα, με σύγκριση ΜΟΝΟ γραμμάτων και ψηφίων.

    ΑΠΟΚΛΙΣΗ 22/11/2026 (μετά το 1ο τρέξιμο, πριν από τις αλλοιώσεις): η σύγκριση με κενά και
    παύλες σημάδευε σωστά χωρία για λόγους μορφής: «anti-pattern» απέναντι σε «anti-⏎pattern»,
    χωρίο κομμένο σε «la-», «1⁄4». Πίνακες: το μοντέλο αντιγράφει κελιά που στο κείμενο της σελίδας
    δεν είναι συνεχόμενα, και τα χωρίζει με αλλαγές γραμμής ή «...». Τότε ΚΑΘΕ κομμάτι πρέπει να
    υπάρχει στη σελίδα, και το μεγαλύτερο να έχει ≥ `min_len` γράμματα/ψηφία.

    2η ΑΠΟΚΛΙΣΗ 22/11/2026 (2ο τρέξιμο, πάλι πριν από τις αλλοιώσεις): το ελάχιστο 15 σημάδεψε 12
    σωστά στοιχεία λίστας εντολής («--good», «--passes=2», q028), που στο 1ο τρέξιμο περνούσαν. Το
    `apply_check` περνά min_len = min(15, μήκος της ίδιας της πρότασης): ένα χωρίο δεν χρειάζεται
    να είναι μεγαλύτερο από αυτό που στηρίζει."""
    page = _alnum(source_text)
    for split in _FRAGMENT_SPLITS:
        frags = [f for f in (_alnum(x) for x in split.split(quote or "")) if f]
        if frags and max(map(len, frags)) >= min_len and all(f in page for f in frags):
            return True
    return False


def quote_cover(quote: str, pages) -> float:
    """Διάγνωση, όχι κρίση: το μεγαλύτερο ΣΥΝΕΧΕΣ κομμάτι του χωρίου (σε γράμματα/ψηφία) που
    υπάρχει σε κάποια σελίδα, ως ποσοστό του χωρίου. ~1.0 = σχεδόν αυτούσιο (λάθος μορφής),
    χαμηλό = ενωμένα μακρινά κομμάτια ή παράφραση."""
    q = _alnum(quote)
    if not q:
        return 0.0
    best = 0
    for page in pages:
        sm = difflib.SequenceMatcher(None, q, _alnum(page), autojunk=False)
        best = max(best, sm.find_longest_match(0, len(q), 0, len(sm.b)).size)
    return round(best / len(q), 2)


# ---------------------------------------------------------------------------
# Αριθμοί (από evidence_cards, library-frozen)
# ---------------------------------------------------------------------------

# Χωρίς έλεγχο ΜΕΤΑ τον αριθμό (απόκλιση 22/11/2026): στο κείμενο των PDF η μονάδα κολλάει
# («538Mbps», «7.5x», «100msincrements», «9.7s») και ο αριθμός χανόταν. Πριν από τον αριθμό
# μένει ο έλεγχος: «S3», «EC2», «t3.nano» δεν είναι αριθμοί.
_NUM_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[-\s](" + "|".join(_UNITS[1:10]) + r"))?\b|\b("
    + "|".join(_UNITS) + r")\b",
    re.IGNORECASE,
)
# Ελληνικά αριθμητικά που γράφει το μοντέλο αντί για ψηφία («δύο», «τρεις»).
_EL_NUMS = {"ένα": 1, "μία": 1, "μια": 1, "δύο": 2, "τρία": 3, "τρεις": 3, "τέσσερα": 4,
            "τέσσερις": 4, "πέντε": 5, "έξι": 6, "επτά": 7, "εφτά": 7, "οκτώ": 8, "οχτώ": 8,
            "εννέα": 9, "εννιά": 9, "δέκα": 10}


def _number_forms(text: str) -> list[tuple[str, frozenset[str]]]:
    """Κάθε αριθμός του κειμένου με τις αποδεκτές μορφές του (δεκαδικό κόμμα/τελεία, χιλιάδες,
    μηδενικά στο τέλος)."""
    t = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    out = []
    for m in _NUM_RE.finditer(t):
        raw = m.group(0)
        forms = {raw.replace(",", ".")}
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", raw):
            forms.add(re.sub(r"[.,]", "", raw))
        forms |= {f.rstrip("0").rstrip(".") for f in list(forms) if "." in f}
        out.append((raw, frozenset(forms)))
    return out


def _word_numbers(text: str) -> set[str]:
    out = set()
    for m in _WORD_NUM_RE.finditer(text or ""):
        if m.group(1):
            n = _TENS[m.group(1).lower()] + (_UNITS.index(m.group(2).lower()) if m.group(2) else 0)
        else:
            n = _UNITS.index(m.group(3).lower())
        out.add(str(n))
    low = (text or "").lower()
    for word, n in _EL_NUMS.items():
        if re.search(rf"(?<!\w){word}(?!\w)", low):
            out.add(str(n))
    return out


def numbers_in(text: str) -> set[str]:
    have: set[str] = set()
    for _raw, forms in _number_forms(text):
        have |= forms
    return have | _word_numbers(text)


def unsupported_numbers(claim: str, cited_texts: list[str], allowed: set[str] | None = None) -> list[str]:
    """Αριθμοί του ισχυρισμού που δεν υπάρχουν σε καμία παραπεμπόμενη σελίδα (ως ψηφία ή ως
    λέξεις). `allowed`: αριθμοί που έρχονται από την ερώτηση (π.χ. έτος άρθρου)."""
    have = set(allowed or ())
    for t in cited_texts:
        have |= numbers_in(t)
    return sorted({raw for raw, forms in _number_forms(claim) if not forms & have})


# ---------------------------------------------------------------------------
# Χωρισμός της απάντησης σε ισχυρισμούς
# ---------------------------------------------------------------------------

_CITE_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_LABEL = re.compile(r"S\d+")
_BULLET = re.compile(r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZΑ-ΩΆ-Ώ\"«])")


@dataclass
class Claim:
    i: int
    text: str                      # χωρίς τις παραπομπές
    labels: list[str]              # π.χ. ["S1", "S3"], με τη σειρά που γράφτηκαν
    kind: str = "claim"            # claim | refusal | lead_in
    absence: bool = False          # λέει τι ΔΕΝ υπάρχει στις πηγές (ταιριάζει με το refusal_re)
    # αποτελέσματα
    numbers_missing: list[str] = field(default_factory=list)
    quote: str | None = None
    quote_label: str | None = None
    quote_found: bool = False      # (β) το χωρίο υπάρχει αυτούσιο στη σελίδα quote_label
    quote_cover: float | None = None  # ΜΟΝΟ διάγνωση όταν δεν βρέθηκε: μέγιστο συνεχές κοινό κομμάτι
    cite_ok: bool = False          # (β) ... και η σελίδα αυτή παραπέμπεται από την πρόταση
    support: str | None = None     # (γ) full | partial | none
    missing: str = ""
    status: str = ""               # supported | flagged | skipped
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _split_line(line: str) -> list[str]:
    parts = [p for p in _SENT_END.split(line) if p.strip()]
    out: list[str] = []
    for p in parts:
        # «… κείμενο. [S1]» -> η παραπομπή ανήκει στην προηγούμενη πρόταση
        if out and not _CITE_GROUP.sub("", p).strip(" .;,"):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def split_claims(answer: str, refusal_re: re.Pattern | None = None) -> list[Claim]:
    """Προτάσεις της απάντησης με τις ετικέτες τους. Οι προτάσεις άρνησης («τα έγγραφα δεν
    αναφέρουν…») και οι εισαγωγές λίστας χωρίς παραπομπή δεν είναι ισχυρισμοί."""
    claims: list[Claim] = []
    text = (answer or "").replace("**", "")
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        for sent in _split_line(line):
            labels = [lab for g in _CITE_GROUP.findall(sent) for lab in _LABEL.findall(g)]
            body = re.sub(r"\s+([.,;:])", r"\1", _CITE_GROUP.sub("", sent)).strip()
            if len(body.split()) < MIN_CLAIM_WORDS and not labels:
                continue
            kind = "claim"
            if not labels and refusal_re is not None and refusal_re.search(body):
                kind = "refusal"        # με παραπομπή = περιέχει και ισχυρισμό -> ελέγχεται
            elif not labels and body.endswith(":"):
                kind = "lead_in"
            absence = bool(refusal_re is not None and refusal_re.search(body))
            claims.append(Claim(i=len(claims) + 1, text=body, labels=list(dict.fromkeys(labels)),
                                kind=kind, absence=absence))
    return claims


# ---------------------------------------------------------------------------
# Κλήση ελέγχου (ΜΙΑ ανά απάντηση)
# ---------------------------------------------------------------------------

# Αγγλικά ΕΠΙΤΗΔΕΣ: τα κείμενα είναι αγγλικά, και κάθε κρίση μοντέλου εδώ τρέχει στη γλώσσα του
# σώματος (11/8/2026: ελληνική ερώτηση πάνω σε αγγλικό κείμενο έκρινε λάθος). Οι ελληνικοί
# ισχυρισμοί μένουν ελληνικοί — η μέτρηση γίνεται ΧΩΡΙΣΤΑ ανά γλώσσα.
CHECK_PROMPT = """You verify a research assistant's answer against the source passages it was given.

For EACH numbered CLAIM below, return:
- "quote": the shortest contiguous span (at most {max_words} words), copied character for character \
from ONE source passage, that supports the claim. Prefer the passages the claim cites. If no passage \
supports any part of it, use null.
- "label": the S-label of the passage the quote comes from (e.g. "S3"), or null.
- "support": "full" if the source passages support EVERY part of the claim: each number, unit, year, \
name, scope (time period, population, setting) and each causal or comparative relation; "partial" if \
they support only some of it; "none" if they support none of it.
- "missing": when partial or none, the part of the claim the sources do not support (a few English \
words); otherwise "".

Claims may be written in Greek while the sources are in English: judge the meaning, not the language.
Judge ONLY against the SOURCE PASSAGES, not your own knowledge. A claim that is true but absent from \
the passages is "none".

Return ONLY JSON, no prose:
{{"claims": [{{"i": 1, "quote": "...", "label": "S1", "support": "full", "missing": ""}}]}}

--- SOURCE PASSAGES ---
{sources}

--- QUESTION (for context only) ---
{question}

--- CLAIMS ---
{claims}

JSON:"""


def label_texts(pages: list[tuple[str, dict]], source_key=None) -> dict[str, str]:
    """S-ετικέτα -> κείμενο, με την ΙΔΙΑ αρίθμηση με το ask_ai: η ετικέτα δίνεται στην πρώτη
    εμφάνιση κάθε (έγγραφο, σελίδα) και οι επαναλήψεις μοιράζονται την ετικέτα της."""
    key_of = source_key or (lambda m: (m.get("doc_id") or m.get("file_name"), m.get("page")))
    labels: dict = {}
    texts: dict[str, list[str]] = {}
    for i, (text, meta) in enumerate(pages, 1):
        lab = labels.setdefault(key_of(meta), f"S{i}")
        texts.setdefault(lab, []).append(text)
    return {lab: "\n".join(v) for lab, v in texts.items()}


def build_check_prompt(question: str, claims: list[Claim], texts: dict[str, str]) -> str | None:
    todo = [c for c in claims if c.kind == "claim"]
    if not todo:
        return None
    sources = "\n\n".join(f"[{lab}]\n{t}" for lab, t in texts.items())
    lines = "\n".join(
        f"{c.i}. (cites {', '.join(c.labels) if c.labels else 'nothing'}) {c.text}" for c in todo)
    return CHECK_PROMPT.format(max_words=MAX_QUOTE_WORDS, sources=sources, question=question,
                               claims=lines)


def parse_check(raw: str) -> dict[int, dict]:
    """{i: {"quote", "label", "support", "missing"}}. Ανέχεται ```json και κείμενο γύρω."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except ValueError:
                obj = None
    items = (obj or {}).get("claims") if isinstance(obj, dict) else obj
    out: dict[int, dict] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            i = int(str(it.get("i")))
        except (TypeError, ValueError):
            continue
        sup = str(it.get("support") or "").strip().lower()
        out[i] = {"quote": it.get("quote") or None,
                  "label": (str(it.get("label")).strip().strip("[]") if it.get("label") else None),
                  "support": sup if sup in ("full", "partial", "none") else None,
                  "missing": str(it.get("missing") or "")}
    return out


def apply_check(claims: list[Claim], texts: dict[str, str], verdicts: dict[int, dict] | None,
                question: str = "") -> list[Claim]:
    """Ντετερμινιστικοί έλεγχοι + η κρίση του μοντέλου -> status ανά πρόταση.

    supported = το μοντέλο λέει full ΚΑΙ το χωρίο βρέθηκε αυτούσιο ΚΑΙ κάθε αριθμός υπάρχει σε
    παραπεμπόμενη σελίδα. Οτιδήποτε άλλο -> flagged, με τον πρώτο λόγο. Η παραπομπή (α) και το
    σωστό σημείο (β) ΔΕΝ μπαίνουν στο status: μετριούνται χωριστά (`labels`, `cite_ok`)."""
    allowed = numbers_in(question)
    for c in claims:
        if c.kind != "claim":
            c.status, c.reason = "skipped", c.kind
            continue
        # χωρίς παραπομπή: οι αριθμοί ελέγχονται σε ΟΛΕΣ τις σελίδες· το (α) μετριέται χωριστά
        cited = [texts[lab] for lab in c.labels if lab in texts] if c.labels else list(texts.values())
        c.numbers_missing = unsupported_numbers(c.text, cited, allowed)
        v = (verdicts or {}).get(c.i)
        min_len = max(4, min(MIN_QUOTE_ALNUM, len(_alnum(c.text))))
        if v:
            c.quote, c.quote_label, c.support, c.missing = (v["quote"], v["label"], v["support"],
                                                            v["missing"])
            if c.quote and c.quote_label in texts:
                c.quote_found = quote_in_source(c.quote, texts[c.quote_label], min_len)
            if c.quote and not c.quote_found:
                # το μοντέλο έγραψε λάθος ετικέτα αλλά σωστό χωρίο: το βρίσκουμε, και το (β) το
                # λέει ως λάθος σημείο αν η σελίδα δεν παραπέμπεται
                for lab, t in texts.items():
                    if quote_in_source(c.quote, t, min_len):
                        c.quote_found, c.quote_label = True, lab
                        break
            if c.quote and not c.quote_found:
                c.quote_cover = quote_cover(c.quote, texts.values())
            c.cite_ok = c.quote_found and c.quote_label in c.labels
        if c.numbers_missing:
            reason = "number_missing"
        elif not v or v["support"] is None:
            reason = "no_verdict"
        elif v["support"] != "full":
            reason = v["support"]
        elif not c.quote:
            reason = "no_quote"
        elif not c.quote_found:
            reason = "quote_not_in_source"
        else:
            reason = ""
        if reason and c.absence:
            # ΑΠΟΚΛΙΣΗ 22/11/2026: «τα κείμενα δεν αναφέρουν Χ [S1]» — μια απουσία δεν έχει χωρίο
            # να παρατεθεί. Στο 1ο τρέξιμο ΟΛΑ τα σημάδια των σωστών αρνήσεων ήταν τέτοια.
            # Μένει στο prompt όπως ήταν (ίδιες κλήσεις), αλλά δεν σημαδεύεται.
            c.status, c.reason = "skipped", f"absence:{reason}"
            continue
        c.status, c.reason = ("supported", "") if not reason else ("flagged", reason)
    return claims
