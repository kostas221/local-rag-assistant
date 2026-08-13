# backend/evaluation/probe_inline_citations.py
"""Γ2 — INLINE CITATIONS: αξίζουν να μπουν;

ΤΙ ΜΕΤΡΑΕΙ (ντετερμινιστικά, ΜΗΔΕΝ judge):
  1. ΜΟΡΦΗ      — πόσες παραπομπές βγήκαν, πόσες παραμορφωμένες.
  2. ΥΠΑΡΞΗ     — κάθε (αρχείο, σελίδα) πρέπει να είναι μέσα στις σελίδες που
                  ΟΝΤΩΣ στάλθηκαν στο Gemini. Ό,τι δεν είναι = ΕΠΙΝΟΗΜΕΝΟ.
  3. ΘΕΜΕΛΙΩΣΗ  — πόσοι όροι της πρότασης υπάρχουν στη σελίδα που επικαλείται,
                  ΕΝΑΝΤΙ του μέσου όρου στις ΥΠΟΛΟΙΠΕΣ σελίδες του ίδιου
                  context. Αυτό είναι το ΠΑΤΩΜΑ ΤΥΧΗΣ της παραπομπής.
                  Αν τα δύο νούμερα είναι ίδια, οι παραπομπές είναι
                  ΔΙΑΚΟΣΜΗΤΙΚΕΣ: αγκύλες χωρίς παρακολούθηση προέλευσης.
  4. ΚΟΣΤΟΣ     — completion_tokens ανά σκέλος. Η έξοδος κοστίζει 8.3x ανά
                  token έναντι της εισόδου, οπότε αυτή είναι η πρώτη αλλαγή
                  του project που τραβάει τον ΑΚΡΙΒΟ μοχλό.

ΠΩΣ ΜΕΝΕΙ ΕΝΤΙΜΟ ΤΟ ΠΕΙΡΑΜΑ:
  • Η ΑΝΑΚΤΗΣΗ ΤΡΕΧΕΙ ΜΙΑ ΦΟΡΑ ανά ερώτηση και περνάει ως `precomputed` ΚΑΙ
    στα δύο σκέλη -> ίδιες σελίδες, ίδιο context, ίδια tokens εισόδου. Ό,τι
    διαφορά μετρηθεί είναι ΜΟΝΟ του prompt γέννησης.
  • Το σκέλος A είναι ο κώδικας παραγωγής ΑΝΕΓΓΙΧΤΟΣ.
  • Το σκέλος B αλλάζει το prompt με monkeypatch στο PERSONA_STYLES, οπότε το
    prompt χτίζεται από τον ΙΔΙΟ κώδικα. ΔΕΝ αντιγράφεται το prompt εδώ — μια
    αντιγραφή θα μετρούσε άλλο σύστημα από αυτό που τρέχει.
    ΠΡΟΣΟΧΗ: ο κανόνας προσαρτάται στο STYLE (κανόνας 2), άρα εμφανίζεται
    ΠΡΙΝ τους κανόνες 3-5. Αν το πείραμα περάσει, στην παραγωγή μπαίνει ως
    κανόνας 6 και ΞΑΝΑΤΡΕΧΕΙ εδώ — η θέση μέσα στο prompt έχει μετρηθεί ότι
    μετράει (βλ. corrective_v2).
  • --production: ΚΑΝΕΝΑ monkeypatch στο prompt. Το σκέλος cite είναι ο
    κώδικας παραγωγής όπως τρέχει, και το base ο ΙΔΙΟΣ κώδικας με
    CITATION_RULE = "" -> prompt ταυτόσημο με το προ-υιοθέτησης. Αυτό είναι
    το τρέξιμο που επικυρώνει την αλλαγή· τα υπόλοιπα ήταν διερεύνηση.

ΓΛΩΣΣΑ — ΜΕΤΡΗΜΕΝΗ ΠΑΓΙΔΑ: οι ελληνικές απαντήσεις έχουν ελληνικούς όρους,
το σώμα είναι αγγλικό. Λεξιλογική επικάλυψη ελληνικής πρότασης με αγγλική
σελίδα είναι ~0 ΚΑΙ στη σωστή ΚΑΙ στις λάθος σελίδες -> μηδενικός
διαχωρισμός. Γι' αυτό στις ελληνικές ερωτήσεις η θεμελίωση υπολογίζεται ΜΟΝΟ
πάνω σε όρους που επιβιώνουν της μετάφρασης (λατινικοί χαρακτήρες, αριθμοί,
ακρωνύμια, ονόματα αρχείων) και οι προτάσεις με <2 τέτοιους όρους
ΠΑΡΑΛΕΙΠΟΝΤΑΙ αντί να μετρηθούν ως αποτυχία.

ΜΟΡΦΗ (--format): μετρήθηκε ότι το 83% της αύξησης tokens είναι ΤΑ ΟΝΟΜΑΤΑ
ΑΡΧΕΙΩΝ, όχι επιπλέον κείμενο. Το `label` βάζει ετικέτες S1..SN στο context
(μέσω του ai_core.SOURCE_HEADER) και ζητάει «[S3, p.12]» -> ίδια πληροφορία,
μικρότερος λογαριασμός. Ο έλεγχος ύπαρξης γίνεται τότε πάνω στην ετικέτα.

Τρέξε:
  docker compose exec backend python evaluation/probe_inline_citations.py \
      --format label --csv evaluation/runs/inline_citations_label.csv
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_core

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_SET = os.path.join(HERE, "golden_set_50.jsonl")
HARD_SET = os.path.join(HERE, "golden_hard_paraphrase.jsonl")

# Ο κανόνας που δοκιμάζεται. Ρητά: ΜΟΝΟ πηγές του SOURCE TEXT, ΧΩΡΙΣ
# βιβλιογραφία στο τέλος (μια λίστα στο τέλος δεν είναι inline citation και
# θα φούσκωνε τα tokens χωρίς να δίνει προέλευση ανά πρόταση).
CITATION_RULE_FULL = (
    "\n\n    CITATIONS: End every factual sentence with its source in square "
    "brackets, copying the file name and the page number from the "
    "[Source: ...] header of the passage you used, like this: "
    "[example-paper.pdf, p.12]. Always write the REAL file name — never a "
    "placeholder. ONE source per bracket: if a sentence draws on two pages, "
    "write two brackets. Cite ONLY sources that appear in the SOURCE TEXT, "
    "and do not add a bibliography at the end."
)

# COMPACT: μετρήθηκε ότι το 83% της αύξησης tokens είναι ΤΑ ΟΝΟΜΑΤΑ ΑΡΧΕΙΩΝ
# (1.13 χαρ/token έναντι 3.57 του κειμένου). Η ετικέτα S1..S8 δοκιμάζει αν το
# κόστος είναι εγγενές ή απλώς επιλογή μορφής.
LABEL_HEADER = "[S{i}: {file}, Page: {page}]"
# Η επικεφαλίδα ΠΡΙΝ την υιοθέτηση των ετικετών. Χρειάζεται ώστε τα παλιά
# τρεξίματα (--format full/label) να παραμένουν αναπαραγώγιμα: εκείνα μετρήθηκαν
# πάνω σε αυτό το context, και ένα probe που αλλάζει σιωπηλά τη βάση σύγκρισής
# του μετράει άλλο σύστημα από εκείνο που κατέγραψε.
LEGACY_HEADER = "[Source: {file}, Page: {page}]"
# BARE: μετρήθηκε ότι η ΕΤΙΚΕΤΑ βγαίνει σωστή 97/97 ενώ ο αριθμός σελίδας
# βγαίνει 10 φορές λάθος — και όχι επειδή το μοντέλο επινοεί, αλλά επειδή
# αντιγράφει τον ΤΥΠΩΜΕΝΟ αριθμό μέσα από τη σελίδα αντί για το `Page:` της
# επικεφαλίδας. Η ετικέτα αρκεί μόνη της: το (αρχείο, σελίδα) το ξέρει ο
# server. Ό,τι δεν ζητάμε από το μοντέλο, δεν μπορεί να βγει λάθος.
CITATION_RULE_BARE = (
    "\n\n    CITATIONS: End every factual sentence with the label of its "
    "source in square brackets, copying it from the [S...: ...] header of "
    "the passage you used, like this: [S3]. Write ONLY the label — no file "
    "name, no page number, no other text inside the brackets. ONE label per "
    "bracket: if a sentence draws on two passages, write two brackets. Cite "
    "ONLY labels that appear in the SOURCE TEXT, and do not add a "
    "bibliography at the end."
)
CITATION_RULE_LABEL = (
    "\n\n    CITATIONS: End every factual sentence with its source label in "
    "square brackets, copying the label and the page number from the "
    "[S...: ...] header of the passage you used, like this: [S3, p.12]. Use "
    "the label EXACTLY as it appears (S1, S2, ...) — never the file name and "
    "never a placeholder. ONE source per bracket: if a sentence draws on two "
    "pages, write two brackets. Cite ONLY labels that appear in the SOURCE "
    "TEXT, and do not add a bibliography at the end."
)

# Ό,τι μοιάζει με αγκύλη παραπομπής. Το «{2,}» αποκλείει τα [1] [2] των
# βιβλιογραφικών αναφορών που υπάρχουν ΜΕΣΑ στις σελίδες του corpus.
CITE_LOOSE_RE = re.compile(r"\[[^\[\]]{2,200}\]")
# ΜΕΣΑ σε μια αγκύλη μπορεί να υπάρχουν ΠΟΛΛΑ ζεύγη: το μοντέλο γράφει
# «[A.pdf, p. 3, A.pdf, p. 6]». Χωρίς αυτό μετριόταν ως ΕΝΑ κακοσχηματισμένο
# όνομα -> ψευδής «επινόηση». Μετρήθηκε στο smoke test.
PAIR_RE = re.compile(r"([^,\[\]]{2,120}?),\s*(?:pp?\.?|page)\s*(\d{1,4})",
                     re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"[<>]")
# fmt=bare: η αγκύλη περιέχει ΜΟΝΟ ετικέτες. Το «^...$» είναι σκόπιμα
# αυστηρό — αν το μοντέλο προσθέσει οτιδήποτε άλλο, θέλω να το δω ως
# παραμορφωμένη μορφή, όχι να το «σώσω» με χαλαρό regex.
BARE_RE = re.compile(r"^\s*S\d{1,2}(?:\s*[,;]\s*S\d{1,2})*\s*$", re.IGNORECASE)
BARE_LABEL_RE = re.compile(r"S\d{1,2}", re.IGNORECASE)

_STOP_WORDS = """the a an of and or to in for with that this these those is are was
were be been being on at by from as it its their our your not but if then than
which who whom whose what when where how why can could may might will would
shall should must do does did done have has had also such other more most some
any each both between within without into over under about across during while
after before either neither they them there here also very much many one two"""
STOP = frozenset(_STOP_WORDS.split())

SENT_SPLIT = re.compile(r"(?<=[.!?;·])\s+|\n+")


def norm(s: str) -> str:
    """NFKD + πεζά + πέταγμα διακριτικών και μη-αλφαριθμητικών.

    Ίδια συνάρτηση με το probe_quote_first.py: ο έλεγχος πρέπει να επιβιώνει
    σε κενά, παύλες και εισαγωγικά που το μοντέλο αναπαράγει αλλιώς.
    """
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9α-ω]+", "", s)


TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-Ώά-ώ][\w\-\./]{3,}|\d+(?:[\.,]\d+)?%?")


def content_tokens(sentence: str, latin_only: bool) -> list:
    """Όροι της πρότασης που έχουν νόημα για λεξιλογικό έλεγχο.

    latin_only=True στις ελληνικές απαντήσεις: κρατά μόνο ό,τι επιβιώνει της
    μετάφρασης (λατινικά, αριθμοί, ακρωνύμια, ονόματα αρχείων).
    """
    out = []
    for t in TOKEN_RE.findall(sentence):
        low = t.lower().strip(".,")
        if low in STOP or len(low) < 4:
            continue
        if latin_only and not re.search(r"[A-Za-z0-9]", t):
            continue
        if latin_only and re.search(r"[Α-Ωα-ωΆ-Ώά-ώ]", t):
            continue
        if low not in out:
            out.append(low)
    return out


def overlap(tokens: list, page_text: str) -> float:
    """Ποσοστό όρων που εμφανίζονται στη σελίδα (κανονικοποιημένα)."""
    if not tokens:
        return float("nan")
    npage = norm(page_text)
    hit = sum(1 for t in tokens if norm(t) and norm(t) in npage)
    return hit / len(tokens)


def sentence_before(text: str, end: int) -> str:
    """Η πρόταση που τερματίζει στη θέση `end` (αρχή της παραπομπής).

    Αν το τελευταίο κομμάτι είναι πολύ φτωχό (π.χ. η παραπομπή ακολουθεί
    τελεία), ενώνει και το προηγούμενο — αλλιώς θα μετρούσαμε κενό.
    """
    parts = [p.strip() for p in SENT_SPLIT.split(text[:end]) if p.strip()]
    if not parts:
        return ""
    sent = parts[-1]
    if len(sent.split()) < 4 and len(parts) >= 2:
        sent = parts[-2] + " " + sent
    return sent


def strip_citations(text: str, fmt: str = "full") -> str:
    """Βγάζει ΜΟΝΟ τις αγκύλες που είναι παραπομπές — όχι κάθε αγκύλη.

    Χρειάζεται και για τη μέτρηση προτάσεων και για τον έλεγχο άρνησης: αλλιώς
    το σκέλος cite θα φαινόταν να έχει άλλο κείμενο από το base.
    """
    hit = BARE_RE.match if fmt == "bare" else PAIR_RE.search
    return CITE_LOOSE_RE.sub(
        lambda m: "" if hit(m.group(0)[1:-1]) else m.group(0), text)


def has_greek(s: str) -> bool:
    return bool(re.search(r"[Ͱ-Ͽἀ-ῼ]", s))


def refused(text: str) -> bool:
    """Άρνηση απάντησης — ελληνικά ή αγγλικά, ΜΕΤΑ την αφαίρεση παραπομπών."""
    t = norm(text)
    needles = ["cannotfind", "donotcontain", "doesnotcontain", "notprovided",
               "δενβρεθηκε", "δενπεριεχ", "δενμπορωναβρω", "δεναναφερ",
               "δενυπαρχ"]
    return any(n in t for n in needles)


def load_subset(n_main: int, extra_hard: list) -> list:
    """Ντετερμινιστική στρωματοποιημένη επιλογή: κατηγορία x γλώσσα.

    ΟΧΙ out_of_corpus: εκεί το gate κόβει ΠΡΙΝ τη γέννηση, οπότε δεν υπάρχει
    απάντηση να φέρει παραπομπές. Το ενδιαφέρον αρνητικό είναι το h016, που
    ΠΕΡΝΑΕΙ το gate με κάλυψη 0% -> ο μόνος μετρημένος τρόπος να δούμε αν ο
    κανόνας παραπομπής σπρώχνει το μοντέλο να επινοήσει πηγή.
    """
    with open(MAIN_SET, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    rows = [r for r in rows if r.get("category") != "out_of_corpus"]
    buckets = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        key = (r.get("category"), "gr" if has_greek(r["question"]) else "en")
        buckets.setdefault(key, []).append(r)

    picked, i = [], 0
    while len(picked) < n_main:
        added = False
        for key in sorted(buckets):
            if i < len(buckets[key]) and len(picked) < n_main:
                picked.append(buckets[key][i])
                added = True
        if not added:
            break
        i += 1

    if extra_hard and os.path.exists(HARD_SET):
        with open(HARD_SET, encoding="utf-8") as fh:
            hard = {json.loads(x)["id"]: json.loads(x) for x in fh if x.strip()}
        for hid in extra_hard:
            if hid in hard:
                picked.append(hard[hid])
    return picked


async def run_arm(question: str, precomputed) -> dict:
    """Μία κλήση του ΠΡΑΓΜΑΤΙΚΟΥ ask_ai. Επιστρέφει κείμενο + tokens + πηγές."""
    text, sources, mtr = [], [], {}
    async for ev in ai_core.ask_ai(question, None, history=None, user_id=None,
                                   precomputed=precomputed):
        if ev["type"] == "text":
            text.append(ev["data"])
        elif ev["type"] == "sources":
            sources = ev["data"]
        elif ev["type"] == "metrics":
            mtr = ev["data"]
    return {"text": "".join(text), "sources": sources, "metrics": mtr}


def file_key(name: str) -> str:
    """Κλειδί αρχείου ΑΝΕΞΑΡΤΗΤΟ από την κατάληξη.

    Το μοντέλο γράφει άλλοτε «excamera-nsdi17.pdf» και άλλοτε
    «excamera-nsdi17». Χωρίς αυτό, ΚΑΘΕ σωστή παραπομπή θα μετριόταν
    επινοημένη — θα μετρούσαμε τον matcher μου, όχι το μοντέλο.
    """
    return norm(re.sub(r"\.pdf$", "", name.strip(), flags=re.IGNORECASE))


def parse_citations(answer: str, fmt: str = "full"):
    """-> (cites, bad_format) όπου cites = [(θέση αγκύλης, αρχείο, σελίδα)].

    fmt=bare: η σελίδα δεν ζητείται καθόλου, οπότε επιστρέφεται κενή και ο
    έλεγχος ύπαρξης γίνεται μόνο στην ετικέτα.
    """
    cites, bad_format = [], 0
    for m in CITE_LOOSE_RE.finditer(answer):
        inner = m.group(0)[1:-1]
        if fmt == "bare":
            if BARE_RE.match(inner):
                for lab in BARE_LABEL_RE.findall(inner):
                    cites.append((m.start(), lab, ""))
            else:
                bad_format += 1
            continue
        pairs = PAIR_RE.findall(inner)
        if not pairs:
            bad_format += 1          # αγκύλη χωρίς αναγνωρίσιμη πηγή
            continue
        for f, p in pairs:
            cites.append((m.start(), f.strip(), p))
    return cites, bad_format


def build_index(pages: list, fmt: str) -> dict:
    """pages: [(ετικέτα, αρχείο, σελίδα, κείμενο)] -> {(κλειδί, σελίδα): κείμενο}.

    ΟΙ ΣΕΛΙΔΕΣ ΠΟΥ ΟΝΤΩΣ ΣΤΑΛΘΗΚΑΝ. Στο fmt=label το κλειδί είναι η ετικέτα
    S1..S8, στο fmt=full το όνομα αρχείου. Ο έλεγχος είναι ο ΙΔΙΟΣ και στις
    δύο περιπτώσεις — αλλάζει μόνο τι ζητάμε από το μοντέλο να αντιγράψει.
    """
    out = {}
    for label, fname, page, txt in pages:
        key = norm(label) if fmt in ("label", "bare") else file_key(fname)
        out[(key, "" if fmt == "bare" else str(page))] = txt
    return out


def analyse(answer: str, index: dict, is_greek: bool, unmatched: list,
            fmt: str = "full") -> dict:
    cites, bad_format = parse_citations(answer, fmt)
    known = {k for k, _ in index}

    fabricated, fab_page, placeholder, cited_keys = 0, 0, 0, set()
    g_cited, g_floor, g_skip = [], [], 0

    for start, fname, page in cites:
        if PLACEHOLDER_RE.search(fname):
            placeholder += 1         # αντέγραψε το «<file name>» του κανόνα
            continue
        key = (file_key(fname), page)
        if key not in index:
            fabricated += 1
            # Διάκριση που αλλάζει τη διάγνωση: ΥΠΑΡΚΤΗ πηγή με σελίδα που ΔΕΝ
            # στάλθηκε (το μοντέλο «θυμάται» αριθμό) έναντι πηγής που δεν
            # υπάρχει καθόλου. Το πρώτο είναι το πιο ύπουλο για τον χρήστη.
            if key[0] in known:
                fab_page += 1
            unmatched.append((f"[{fname}, p.{page}]", key, sorted(index)))
            continue
        cited_keys.add(key)
        sent = strip_citations(sentence_before(answer, start), fmt)
        toks = content_tokens(sent, latin_only=is_greek)
        if len(toks) < 2:
            g_skip += 1
            continue
        g_cited.append(overlap(toks, index[key]))
        others = [t for k, t in index.items() if k != key]
        if others:
            g_floor.append(sum(overlap(toks, t) for t in others) / len(others))

    sents = [s for s in SENT_SPLIT.split(strip_citations(answer, fmt))
             if s.strip()]
    return {
        "chars": len(answer),
        "n_cites": len(cites),
        "n_bad_format": bad_format,
        "n_fabricated": fabricated,
        "n_fab_page": fab_page,
        "n_placeholder": placeholder,
        "cited_pages": len(cited_keys),
        "pages_sent": len(index),
        "sentences": len(sents),
        "ground_n": len(g_cited),
        "ground_skipped": g_skip,
        "ground_cited": (sum(g_cited) / len(g_cited)) if g_cited else float("nan"),
        "ground_floor": (sum(g_floor) / len(g_floor)) if g_floor else float("nan"),
        "refused": int(refused(strip_citations(answer, fmt))),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="ερωτήσεις από το κύριο σετ (στρωματοποιημένα)")
    ap.add_argument("--hard", default="h016",
                    help="επιπλέον ids από το hard set, χωρισμένα με κόμμα ('' για κανένα)")
    ap.add_argument("--format", choices=["full", "label", "bare"],
                    default="full", dest="fmt",
                    help="full = [αρχείο.pdf, p.N] · label = [S3, p.N] · "
                         "bare = [S3]")
    ap.add_argument("--csv", default="")
    ap.add_argument("--production", action="store_true",
                    help="σκέλος cite = ο ΚΩΔΙΚΑΣ ΠΑΡΑΓΩΓΗΣ ανέγγιχτος, "
                         "base = ο ΙΔΙΟΣ κώδικας με CITATION_RULE=''")
    args = ap.parse_args()

    prod_rule = getattr(ai_core, "CITATION_RULE", "")
    if args.production:
        # ΓΙΑΤΙ ΞΑΝΑΤΡΕΧΕΙ: το προηγούμενο πείραμα κόλλαγε τον κανόνα μέσα στο
        # STYLE (κανόνας 2), δηλαδή ΠΡΙΝ τον κανόνα μη-ψευδαίσθησης. Στην
        # παραγωγή μπήκε ως κανόνας 6, και η ΘΕΣΗ μέσα στο prompt έχει μετρηθεί
        # ότι μετράει (corrective_v2: ίδιο περιεχόμενο, άλλη θέση, χειρότερο
        # αποτέλεσμα σε κάθε κατώφλι). Εδώ κανένα σκέλος δεν αντιγράφει prompt.
        args.fmt = "bare"
        assert ai_core.SOURCE_HEADER.format(i=1, file="a.pdf", page="7") == \
            "[S1: a.pdf, Page: 7]", "το SOURCE_HEADER δεν έχει ετικέτες S1..SN"
        assert "[S3]" in prod_rule, "ο CITATION_RULE της παραγωγής δεν είναι ο bare"
        base_header, base_rule = ai_core.SOURCE_HEADER, prod_rule
    else:
        # CONTROL: τα σκέλη full/label μετρήθηκαν ΠΡΙΝ μπουν οι ετικέτες στην
        # παραγωγή. Επαναφέρουμε ρητά εκείνη τη βάση, αλλιώς η σύγκριση με τα
        # καταγεγραμμένα νούμερα είναι άκυρη.
        base_header, base_rule = LEGACY_HEADER, ""
        ai_core.SOURCE_HEADER, ai_core.CITATION_RULE = LEGACY_HEADER, ""

    rule = {"label": CITATION_RULE_LABEL, "bare": CITATION_RULE_BARE}.get(
        args.fmt, CITATION_RULE_FULL)

    extra = [x.strip() for x in args.hard.split(",") if x.strip()]
    subset = load_subset(args.n, extra)
    print(f"Ερωτήσεις: {len(subset)} "
          f"({sum(1 for r in subset if has_greek(r['question']))} ελληνικές) "
          f"x 2 σκέλη = {2 * len(subset)} κλήσεις γέννησης")
    print(f"Μορφή παραπομπής: {args.fmt}"
          + ("  (ετικέτες S1..SN στο context)"
             if args.fmt in ("label", "bare") else "")
          + "\n")

    base_style = ai_core.PERSONA_STYLES["Researcher"]
    rows, unmatched = [], []

    for r in subset:
        q, qid = r["question"], r["id"]
        is_gr = has_greek(q)
        # ΜΙΑ ανάκτηση, ΔΥΟ σκέλη -> ίδιο context, ίδια tokens εισόδου.
        precomputed = await ai_core.search_documents(q, None, user_id=None)
        # Η αρίθμηση είναι Η ΙΔΙΑ με το enumerate(top_3_data, 1) του ai_core.
        pages = [(f"S{i}", m.get("file_name", "?"), str(m.get("page", "?")), t)
                 for i, (t, m) in enumerate(precomputed, 1)]

        for arm in ("base", "cite"):
            if args.production:
                # cite: ΤΙΠΟΤΑ δεν πειράζεται. base: σβήνει ΜΟΝΟ ο κανόνας 6 —
                # το context κρατάει τις ίδιες ετικέτες, άρα η ΜΟΝΗ διαφορά
                # των δύο σκελών είναι ο κανόνας, όχι η μορφή της επικεφαλίδας.
                ai_core.CITATION_RULE = prod_rule if arm == "cite" else ""
            elif arm == "cite":
                ai_core.PERSONA_STYLES["Researcher"] = base_style + rule
                if args.fmt in ("label", "bare"):
                    ai_core.SOURCE_HEADER = LABEL_HEADER
            else:
                ai_core.PERSONA_STYLES["Researcher"] = base_style
            try:
                res = await run_arm(q, precomputed)
            finally:
                ai_core.PERSONA_STYLES["Researcher"] = base_style
                ai_core.SOURCE_HEADER = base_header
                ai_core.CITATION_RULE = base_rule

            # Στο base σκέλος το context ΔΕΝ έχει ετικέτες, οπότε ο έλεγχος
            # γίνεται πάντα με ονόματα αρχείων εκεί.
            fmt = args.fmt if arm == "cite" else "full"
            index = build_index(pages, fmt)
            a = analyse(res["text"], index, is_gr,
                        unmatched if arm == "cite" else [], fmt)
            a.update({
                "id": qid, "arm": arm, "fmt": args.fmt,
                "lang": "gr" if is_gr else "en",
                "category": r.get("category", "?"),
                "prompt_tokens": res["metrics"].get("prompt_tokens"),
                "completion_tokens": res["metrics"].get("completion_tokens"),
                "generation_s": res["metrics"].get("generation_s"),
                "answer": res["text"],
            })
            rows.append(a)
            print(f"{qid:5s} {arm:5s} {a['chars']:5d} χαρ · "
                  f"{a['completion_tokens'] or '?':>5} tok · "
                  f"cites {a['n_cites']:2d} (επινοημ. {a['n_fabricated']}, "
                  f"placeholder {a['n_placeholder']}, μορφή {a['n_bad_format']}) · "
                  f"σελ {a['cited_pages']}/{a['pages_sent']} · "
                  f"θεμ {a['ground_cited']:.2f} vs πάτωμα {a['ground_floor']:.2f}"
                  + ("  ΑΡΝΗΣΗ" if a["refused"] else ""))

    # ---------------- ΠΟΡΙΣΜΑ ----------------
    def sel(arm, key, only=None):
        v = [r[key] for r in rows if r["arm"] == arm
             and (only is None or r["lang"] == only)
             and isinstance(r[key], (int, float)) and r[key] == r[key]]
        return v

    def avg(v):
        return sum(v) / len(v) if v else float("nan")

    print("\n--- ΚΟΣΤΟΣ (η έξοδος κοστίζει 8.3x ανά token) ---")
    for arm in ("base", "cite"):
        print(f"  {arm:5s} χαρ {avg(sel(arm, 'chars')):7.0f} · "
              f"tokens εξόδου {avg(sel(arm, 'completion_tokens')):7.0f} · "
              f"tokens εισόδου {avg(sel(arm, 'prompt_tokens')):7.0f}")
    b, c = avg(sel("base", "completion_tokens")), avg(sel("cite", "completion_tokens"))
    if b == b and c == c and b:
        print(f"  Δ εξόδου: {c - b:+.0f} tokens ({100 * (c - b) / b:+.1f}%)")
    gb, gc = avg(sel("base", "generation_s")), avg(sel("cite", "generation_s"))
    if gb == gb and gc == gc:
        print(f"  χρόνος γέννησης: {gb:.2f}s -> {gc:.2f}s ({gc - gb:+.2f}s)")

    print("\n--- ΠΑΡΑΠΟΜΠΕΣ (σκέλος cite) ---")
    tot = sum(sel("cite", "n_cites"))
    fab = sum(sel("cite", "n_fabricated"))
    bad = sum(sel("cite", "n_bad_format"))
    sent = sum(sel("cite", "sentences"))
    print(f"  σύνολο {tot} σε {sent} προτάσεις "
          f"({100 * tot / sent if sent else 0:.0f}% πυκνότητα)")
    fabp = sum(sel("cite", "n_fab_page"))
    print(f"  ΕΠΙΝΟΗΜΕΝΕΣ (σελίδα εκτός context): {fab}"
          + (f"  (από αυτές {fabp} με ΥΠΑΡΚΤΗ πηγή, λάθος σελίδα)" if fab else "")
          + ("   <-- ΣΚΛΗΡΗ ΑΠΟΤΥΧΙΑ" if fab else "   <-- καθαρό"))
    if unmatched:
        # ΔΙΑΓΝΩΣΤΙΚΟ: πριν πιστέψεις «επινοημένη», δες αν φταίει ο matcher.
        print("  --- ασύμφωνες παραπομπές (πρώτες 6) ---")
        for raw, key, keys in unmatched[:6]:
            print(f"    {raw}  -> {key}")
            print(f"      διαθέσιμα: {keys}")
    print(f"  placeholder (αντέγραψε το πρότυπο): "
          f"{sum(sel('cite', 'n_placeholder'))}")
    print(f"  παραμορφωμένη μορφή: {bad}")
    print(f"  σελίδες που επικαλείται: {avg(sel('cite', 'cited_pages')):.1f} "
          f"από {avg(sel('cite', 'pages_sent')):.1f} που στάλθηκαν")

    print("\n--- ΘΕΜΕΛΙΩΣΗ ΕΝΑΝΤΙ ΠΑΤΩΜΑΤΟΣ ΤΥΧΗΣ ---")
    for lang in (None, "en", "gr"):
        gc, gf = sel("cite", "ground_cited", lang), sel("cite", "ground_floor", lang)
        if not gc:
            continue
        label = {None: "ΟΛΕΣ", "en": "αγγλικές", "gr": "ελληνικές"}[lang]
        d = avg(gc) - avg(gf)
        print(f"  {label:9s} επικαλούμενη {avg(gc):.3f} · "
              f"άλλες σελίδες {avg(gf):.3f} · Δ {d:+.3f}"
              + ("   ΔΙΑΚΟΣΜΗΤΙΚΕΣ" if d < 0.05 else "   πραγματική προέλευση"))
    print(f"  προτάσεις που παραλείφθηκαν (λίγοι όροι): "
          f"{sum(sel('cite', 'ground_skipped'))}")

    print("\n--- ΑΡΝΗΣΕΙΣ (2η γραμμή άμυνας) ---")
    for arm in ("base", "cite"):
        ref = [r["id"] for r in rows if r["arm"] == arm and r["refused"]]
        print(f"  {arm:5s}: {len(ref)} {ref}")
    h = [r for r in rows if r["id"] == "h016"]
    if h:
        print("  h016 (κάλυψη 0%, περνάει το gate): "
              + " · ".join(f"{r['arm']}={'ΑΡΝΗΣΗ' if r['refused'] else 'ΑΠΑΝΤΑ'}"
                           f" cites={r['n_cites']} επιν={r['n_fabricated']}"
                           for r in h))

    print("\n--- ΚΑΤΗΓΟΡΙΕΣ ---")
    print("  " + str(Counter(r["category"] for r in rows if r["arm"] == "cite")))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        cols = ["id", "arm", "fmt", "lang", "category", "chars", "prompt_tokens",
                "completion_tokens", "generation_s", "sentences", "n_cites",
                "n_bad_format",
                "n_fabricated", "n_fab_page", "n_placeholder", "cited_pages",
                "pages_sent", "ground_n",
                "ground_skipped", "ground_cited", "ground_floor", "refused",
                "answer"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
