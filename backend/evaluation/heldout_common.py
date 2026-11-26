"""Κοινά του βήματος 7 (τελική μέτρηση στο κλειστό σετ, docs/HELDOUT_SET_PROTOCOL.md §4, §4α).

ΜΟΝΟ stdlib: το διαβάζουν και το `run_heldout.py` (Docker) και το `judge_heldout.py` / το
`heldout_preflight.py` (host, Git Bash), και ελέγχεται στο fast CI.

ΚΑΝΟΝΑΣ ΕΞΟΔΟΥ: κανένα script του βήματος 7 δεν τυπώνει κείμενο ερώτησης, απάντησης αναφοράς ή
απάντησης του συστήματος. Η έξοδός τους επικολλάται στη συνομιλία ανάπτυξης, που δεν διαβάζει ποτέ το
περιεχόμενο του κλειστού σετ. Τα κείμενα γράφονται μόνο στο `runs/heldout*/private/` (εκτός git).
"""
import csv
import hashlib
import json
import math
import os
import random
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)

# Το σύστημα που μετριέται: τα αρχεία του backend (εκτός evaluation/ και tests/) όπως στο commit αυτό.
FREEZE_COMMIT = "c15a395"

MODES = {
    "heldout": {
        "csv": os.path.join(HERE, "heldout_questions_draft.csv"),
        "sources": os.path.join(HERE, "heldout_sources.json"),
        "papers": os.path.join(HERE, "test_papers", "heldout"),
        "out": os.path.join(HERE, "runs", "heldout"),
        # Κλείδωμα της 19/11/2026 (HELDOUT_SET_PROTOCOL.md §3α)
        "csv_sha": "b54e663c307f33e9f3c57ff8e32024d85421a8e5723903082db619ac210795e1",
        "sources_sha": "7f6dc0394828a0bee1b217207afe0bd2fffc2ffe18a299ad37df3275a4f54adb",
    },
    # Δοκιμή του εργαλείου σε ερωτήσεις ΑΝΑΠΤΥΞΗΣ (άρθρα cloud), ποτέ στο κλειστό σετ.
    "smoke": {
        "csv": os.path.join(HERE, "heldout_smoke_questions.csv"),
        "sources": None,
        "papers": os.path.join(HERE, "test_papers", "cloud"),
        "out": os.path.join(HERE, "runs", "heldout_smoke"),
        "csv_sha": None,
        "sources_sha": None,
    },
}

# Οι ρυθμίσεις του παγωμένου συστήματος (defaults του ai_core στο c15a395). Το run_heldout τις συγκρίνει
# με τις τιμές που ισχύουν ΠΡΑΓΜΑΤΙΚΑ (το .env μπορεί να τις αλλάζει) και σταματά σε κάθε διαφορά.
FROZEN_CONFIG = {
    "GEMINI_MODEL": "gemini-3.8-flash",
    "TEMPERATURE": 0.1,
    "MAX_OUTPUT_TOKENS": 4096,
    "THINKING_BUDGET": 512,
    "USE_REST_GENERATION": True,
    "EMBED_MODEL_NAME": "BAAI/bge-m3",
    "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "MIN_RERANK_SCORE": -2.6,
    "CORRECTIVE_MIN_SCORE": -3.8,
    "ENABLE_CORRECTIVE": True,
    "ENABLE_SUFFICIENCY_VETO": True,
    "STRIP_HIDDEN_TEXT": False,
    "ENABLE_FULL_TEXT": True,
    "FULL_TEXT_MAX_TOKENS": 70000,
    "ENABLE_PERDOC": True,
    "PERDOC_EXTRA": 4,
    "RERANK_CANDIDATES": 15,
    "DENSE_CANDIDATES": 30,
    "EXPAND_INPUT": 12,
    "MAX_PAGES": 8,
    "PAGE_SCORE_TOP_K": 1,
    "MAX_PAGES_PER_FILE": 8,
    "USE_BGE_SPARSE": False,
    "CITATION_RULE_SET": True,
}

# Τα κείμενα που βάζει η ask_ai όταν ΑΠΟΤΥΧΕΙ η κλήση του Gemini (σφάλμα API, όχι απάντηση του μοντέλου).
# Χωρίς αυτόν τον έλεγχο ένα 429 θα μετρούσε ως λάθος απάντηση του συστήματος.
API_ERROR_TEXTS = ("Temporary AI issue (possibly a rate limit)", "Προσωρινό πρόβλημα με το AI")

TYPES = ("single", "two_papers", "near", "table")
# Οι 5 κατηγορίες αναφοράς και ανθρώπινου ελέγχου (το two_papers μετριέται σε δύο σκέλη).
CATEGORIES = ("single", "two_papers_a", "two_papers_b", "near", "table")
ARMS = ("alpha", "beta", "closed")
REVIEW_SEED = 20261123
REVIEW_PER_CATEGORY = 2
MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------------------------- αρχεία

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def private_dir(mode: str) -> str:
    return os.path.join(MODES[mode]["out"], "private")


def write_json_atomic(path: str, data) -> None:
    """tmp + os.replace, με λίγες επαναλήψεις: στα Windows το antivirus κλειδώνει στιγμιαία αρχεία."""
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def _pdf(name: str) -> str:
    name = (name or "").strip()
    return name if not name or name.lower().endswith(".pdf") else name + ".pdf"


def _int(value: str):
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


def load_items(path: str) -> list[dict]:
    """Το CSV του κλειστού σετ (ή του smoke) σε ενιαία μορφή. Δεν τυπώνει τίποτα."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        items.append({
            "id": r["id"].strip(),
            "type": r["τύπος"].strip(),
            "lang": r["γλώσσα"].strip(),
            "doc1": _pdf(r["άρθρο_1"]),
            "doc2": _pdf(r.get("άρθρο_2", "")) or None,
            "question": r["ερώτηση"].strip(),
            "reference": r["απάντηση_αναφοράς"].strip(),
            "page1": _int(r.get("σελίδα_1", "")),
            "quote1": (r.get("απόσπασμα_1") or "").strip(),
            "page2": _int(r.get("σελίδα_2", "")),
            "quote2": (r.get("απόσπασμα_2") or "").strip(),
            "missing": (r.get("τι_λείπει") or "").strip(),
            "behaviour": (r.get("σωστή_συμπεριφορά") or "").strip(),
            "values": parse_table_values(r.get("τιμές_πίνακα", "")),
        })
    return items


def category_of(item_type: str, arm: str) -> str | None:
    if arm == "closed":
        return None
    if item_type == "two_papers":
        return "two_papers_a" if arm == "alpha" else "two_papers_b"
    return item_type


# ------------------------------------------------------------------------------------- τιμές πινάκων

_PLAIN = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def parse_table_values(text: str) -> list[dict]:
    """`τιμή · μονάδα · γραμμή/στήλη; ...` -> λίστα. `value` μόνο για ΑΠΛΟΥΣ αριθμούς (π.χ. 0.95, -3,
    12): εύρη, εκθέτες, ± κ.λπ. μένουν None και κρίνονται από τον κριτή και υποχρεωτικά από άνθρωπο."""
    out = []
    for part in (text or "").split(";"):
        if not part.strip():
            continue
        fields = [x.strip() for x in part.split("·")]
        raw = fields[0] if fields else ""
        val = raw.replace("−", "-").replace(" ", "")
        plain = bool(_PLAIN.match(val))
        out.append({
            "raw": raw,
            "value": float(val) if plain else None,
            "unit": fields[1] if len(fields) > 1 else "",
            "label": " · ".join(fields[2:]) if len(fields) > 2 else "",
        })
    return out


_CITATION = re.compile(r"\[S\d+\]")
_NUM = re.compile(r"(?<![\w.,])([-+]?)(\d+(?:\.\d+)?)(?!\d)")
_SPACE_THOUSANDS = re.compile(r"(?<![\w.])(\d{1,3}) (?=\d{3}(?!\d))")


def _normalize_numbers_text(text: str) -> str:
    s = _CITATION.sub(" ", text or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("−", "-").replace("–", "-").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\*\*|__|`", "", s)


def answer_numbers(text: str) -> set[float]:
    """Όλοι οι αριθμοί της απάντησης, με πρόσημο, ως ΟΛΟΚΛΗΡΑ tokens (το 0.5 δεν βρίσκεται μέσα στο
    10.5). Διαβάζεται με ΔΥΟ συμβάσεις και κρατιέται η ένωση, γιατί οι απαντήσεις είναι και ελληνικές:
      · αγγλική: «29,423» = 29423, «0.423» = 0,423·
      · ελληνική: «0,423» = 0,423, «1.000» = 1000.
    Το «[S3]» είναι παραπομπή, όχι ο αριθμός 3."""
    s = _normalize_numbers_text(text)
    variants = []
    for merge_spaces in (False, True):
        base = _SPACE_THOUSANDS.sub(r"\1", s) if merge_spaces else s
        en = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", base)
        en = re.sub(r"(?<=\d),(?=\d)", ".", en)
        el = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", base)
        el = re.sub(r"(?<=\d),(?=\d)", ".", el)
        variants += [en, el]
    values = set()
    for t in variants:
        for m in _NUM.finditer(t):
            v = float(m.group(2))
            values.add(-v if m.group(1) == "-" else v)
    return values


def value_found(value: float, numbers: set[float]) -> bool:
    """Αριθμητική ισότητα με πρόσημο: 22 == 22.0, αλλά 0.5 != -0.5 και 0.5 != 10.5."""
    return any(abs(v - value) <= 1e-9 * max(1.0, abs(value)) for v in numbers)


def check_table_values(values: list[dict], answer: str) -> dict:
    """{'plain_total', 'plain_found', 'plain_ok', 'non_plain'} — ο αυτόματος, αυστηρός έλεγχος τιμής."""
    nums = answer_numbers(answer)
    plain = [v for v in values if v["value"] is not None]
    found = [v["raw"] for v in plain if value_found(v["value"], nums)]
    return {"plain_total": len(plain), "plain_found": len(found),
            "plain_ok": len(found) == len(plain),
            "non_plain": sum(v["value"] is None for v in values)}


# ------------------------------------------------------------------------------------------- στατιστικά

def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    """95 % διάστημα Wilson για k/n (None, None αν n = 0)."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def percentile(values: list[float], q: float) -> float | None:
    """Γραμμική παρεμβολή (όπως numpy 'linear'). q στο [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# ------------------------------------------------------------------------------- κανόνες επιτυχίας

def decide(item: dict, arm: str, row: dict, j: dict) -> dict:
    """Σωστό / όχι για ΜΙΑ απάντηση, με τους κανόνες της προκαταχώρισης.

    j: οι κρίσεις — 'main' (correct|partial|refused|wrong), 'mh' ({'paper_1','paper_2'}), 'near'
    (correct|soft_leak|leak), 'table' (αποτέλεσμα check_table_values).
    Επιστρέφει {'correct': bool | None, 'why': str}; None = τεχνική αποτυχία (ορατή χωριστά)."""
    if row.get("status") != "ok":
        return {"correct": None, "why": "technical_failure"}
    t = item["type"]
    if t == "single":
        if j.get("main") != "correct":
            return {"correct": False, "why": f"judge:{j.get('main')}"}
        if arm != "closed" and not row.get("cited"):
            return {"correct": False, "why": "no_citation"}
        return {"correct": True, "why": ""}
    if t == "two_papers":
        mh = j.get("mh") or {}
        ok = mh.get("paper_1") == "correct" and mh.get("paper_2") == "correct"
        return {"correct": ok, "why": "" if ok else f"judge:{mh.get('paper_1')}/{mh.get('paper_2')}"}
    if t == "near":
        if item["behaviour"] == "refuse":
            if row.get("outcome") == "cut":
                return {"correct": True, "why": "guard_silence"}
            ok = j.get("near") == "correct"
            return {"correct": ok, "why": "" if ok else f"near:{j.get('near')}"}
        ok = j.get("near") == "correct" and j.get("main") == "correct"
        return {"correct": ok, "why": "" if ok else f"near:{j.get('near')} main:{j.get('main')}"}
    if t == "table":
        tv = j.get("table") or {}
        if not tv.get("plain_ok", False):
            return {"correct": False, "why": f"value:{tv.get('plain_found')}/{tv.get('plain_total')}"}
        ok = j.get("main") == "correct"
        return {"correct": ok, "why": "" if ok else f"judge:{j.get('main')}"}
    raise ValueError(f"άγνωστος τύπος {t}")


def false_refusal(item: dict, row: dict, j: dict) -> bool:
    """Άρνηση σε ερώτηση που ΕΧΕΙ απάντηση (single, two_papers, table)."""
    if item["type"] not in ("single", "two_papers", "table") or row.get("status") != "ok":
        return False
    if row.get("outcome") == "cut":
        return True
    if item["type"] == "two_papers":
        mh = j.get("mh") or {}
        return "declared_missing" in (mh.get("paper_1"), mh.get("paper_2"))
    return j.get("main") == "refused"


def critical_candidates(item: dict, arm: str, row: dict, j: dict) -> list[str]:
    """Υποψήφια κρίσιμα λάθη (§4α). Τα επιβεβαιώνει ή τα απορρίπτει ο άνθρωπος, στα PDF.
      c1 ισχυρισμός χωρίς στήριξη: χωρίο που δεν βρέθηκε, αριθμός που λείπει από την παρατιθέμενη
         σελίδα, ή «wrong» από τον κριτή·
      c2 λάθος τιμή σε πίνακα (όχι απλώς τιμή που λείπει): αποτυχία τιμής/κριτή χωρίς άρνηση·
      c3 επινόηση σε ερώτηση άρνησης: κριτής near «leak»·
      c4 παραπομπή σε λάθος άρθρο: χωρίο βρέθηκε σε σελίδα άλλου άρθρου από αυτό που παρατίθεται, ή
         «wrong» του κριτή δύο άρθρων (απόδοση σε λάθος άρθρο)."""
    if arm == "closed" or row.get("status") != "ok":
        return []
    out = []
    p = row.get("passages") or {}
    items = p.get("items") or []
    if any(x.get("status") == "not_found" for x in items) or p.get("numbers_missing"):
        out.append("c1")
    if j.get("main") == "wrong" and "c1" not in out:
        out.append("c1")
    if item["type"] == "table" and row.get("outcome") != "cut" and j.get("main") != "refused":
        tv = j.get("table") or {}
        if not tv.get("plain_ok", True) or j.get("main") in ("wrong", "partial"):
            out.append("c2")
    if item["type"] == "near" and item["behaviour"] == "refuse" and j.get("near") == "leak":
        out.append("c3")
    cited_files = {s["file"] for s in row.get("sources") or [] if s.get("label") in set(row.get("cited") or [])}
    if any(x.get("status") == "found_elsewhere" and x.get("file") and x["file"] not in cited_files
           for x in items):
        out.append("c4")
    mh = j.get("mh") or {}
    if "wrong" in (mh.get("paper_1"), mh.get("paper_2")) and "c4" not in out:
        out.append("c4")
    return out


def review_flags(item: dict, arm: str, row: dict, correct: bool | None = None) -> list[str]:
    """Λόγοι για υποχρεωτικό ανθρώπινο έλεγχο εκτός από τα κρίσιμα λάθη.

    other_paper_cited ΜΟΝΟ όταν η απάντηση δεν κρίθηκε σωστή: υπάρχει για να πιάσει τη θεμιτή απάντηση από
    άλλο άρθρο που ο κριτής μέτρησε λάθος ή επινόηση (λάθος του σετ). Στη δοκιμή του εργαλείου χτυπούσε και
    σε σωστές αρνήσεις που ανέφεραν σχετικό υλικό άλλου άρθρου, όπου δεν έχει τίποτα να πιάσει."""
    if arm == "closed" or row.get("status") != "ok":
        return []
    flags = []
    if row.get("other_papers_cited") and correct is not True:
        flags.append("other_paper_cited")      # ίσως σωστή απάντηση από άλλο άρθρο (λάθος του σετ)
    if item["type"] == "table" and any(v["value"] is None for v in item["values"]):
        flags.append("non_plain_value")
    if (row.get("passages") or {}).get("status") not in ("done", "no_pages"):
        flags.append("passage_check_missing")
    return flags


def sample_random(pool_by_category: dict, seed: int = REVIEW_SEED,
                  per_category: int = REVIEW_PER_CATEGORY) -> list[str]:
    """Οι τυχαίες ερωτήσεις του ανθρώπινου ελέγχου: `per_category` από κάθε κατηγορία, ΜΟΝΟ από όσες
    δεν σημαδεύτηκαν (οι σημαδεμένες ελέγχονται όλες). Ντετερμινιστικό για ίδιο pool και seed."""
    rng = random.Random(seed)
    out = []
    for cat in CATEGORIES:
        pool = sorted(pool_by_category.get(cat, []))
        out += rng.sample(pool, min(per_category, len(pool)))
    return out


# ------------------------------------------------------------------------------- αποτύπωμα συστήματος

def is_system_file(rel: str) -> bool:
    """Αρχείο του συστήματος = κάθε αρχείο του backend εκτός από evaluation/ και tests/."""
    rel = rel.replace("\\", "/")
    return rel.startswith("backend/") and not rel.startswith(("backend/evaluation/", "backend/tests/"))
