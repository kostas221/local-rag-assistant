"""Έλεγχος του κλειστού σετ (heldout_questions_draft.csv) ΧΩΡΙΣ να τρέξει το σύστημα. Μόνο stdlib.

Τρέχει ο ίδιος όποιος γράφει τις ερωτήσεις, ώστε να διορθώσει τα λάθη του πριν κλειδώσει το σετ:

    python backend/evaluation/check_heldout_questions.py

ΣΦΑΛΜΑΤΑ (πρέπει να διορθωθούν): κενό υποχρεωτικό πεδίο, λάθος τύπος/τιμή, σελίδα που δεν υπάρχει,
απόσπασμα που ΔΕΝ βρίσκεται αυτούσιο στη δηλωμένη σελίδα, τιμή πίνακα που δεν υπάρχει στη σελίδα,
ελληνική ερώτηση χωρίς ελληνικά, απόσπασμα > 25 λέξεις.
ΠΡΟΕΙΔΟΠΟΙΗΣΕΙΣ (ξανακοίτα): αγγλική ερώτηση που αντιγράφει ≥ 5 συνεχόμενες λέξεις από τη σελίδα της
(κάνει την αναζήτηση τεχνητά εύκολη), ίδιες ερωτήσεις.
Κείμενα: backend/evaluation/test_papers/heldout_text/<αρχείο>.txt με γραμμές «##### <αρχείο>.pdf pN».
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "heldout_questions_draft.csv"
TEXT_DIR = HERE / "test_papers" / "heldout_text"
TYPES = {"single", "two_papers", "near", "table"}
BEHAVIOURS = {"answer", "partial", "refuse"}
MAX_QUOTE_WORDS = 25
COPY_NGRAM = 5


def norm(s: str) -> str:
    """NFKC, πεζά, ενωμένες λέξεις που έσπασαν με παύλα στο τέλος γραμμής, ενιαία κενά."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def load_pages() -> dict:
    pages = {}
    for f in sorted(TEXT_DIR.glob("*.txt")):
        cur = None
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^##### (\S+\.pdf) p(\d+)\s*$", line)
            if m:
                cur = (m.group(1), int(m.group(2)))
                pages[cur] = []
            elif cur:
                pages[cur].append(line)
    return {k: "\n".join(v) for k, v in pages.items()}


def has_greek(s: str) -> bool:
    return bool(re.search(r"[Ͱ-Ͽἀ-῿]", s or ""))


def copied_ngram(question: str, page: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", norm(question))
    text = " " + " ".join(re.findall(r"[a-z0-9]+", norm(page))) + " "
    for i in range(len(words) - COPY_NGRAM + 1):
        gram = " ".join(words[i:i + COPY_NGRAM])
        if f" {gram} " in text:
            return gram
    return None


def main() -> int:
    if not TEXT_DIR.exists():
        print(f"!! Λείπει ο φάκελος {TEXT_DIR}")
        return 2
    pages = load_pages()
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors, warnings = [], []
    seen = {}

    def page_text(rid, paper, page_s, slot):
        try:
            p = int(str(page_s).strip())
        except ValueError:
            errors.append(f"{rid}: σελίδα_{slot} «{page_s}» δεν είναι αριθμός")
            return None
        key = (paper, p)
        if key not in pages:
            errors.append(f"{rid}: δεν υπάρχει σελίδα {p} στο {paper}")
            return None
        return pages[key]

    def check_quote(rid, quote, text, slot):
        if not quote.strip():
            errors.append(f"{rid}: κενό απόσπασμα_{slot}")
            return
        if len(quote.split()) > MAX_QUOTE_WORDS:
            errors.append(f"{rid}: απόσπασμα_{slot} έχει {len(quote.split())} λέξεις (> {MAX_QUOTE_WORDS})")
        if norm(quote) not in norm(text):
            errors.append(f"{rid}: το απόσπασμα_{slot} ΔΕΝ βρέθηκε αυτούσιο στη δηλωμένη σελίδα: «{quote[:70]}»")

    for r in rows:
        rid, t, lang = r["id"], r["τύπος"], r["γλώσσα"]
        q, ref = r["ερώτηση"].strip(), r["απάντηση_αναφοράς"].strip()
        if t not in TYPES:
            errors.append(f"{rid}: άγνωστος τύπος «{t}»")
            continue
        if not q:
            errors.append(f"{rid}: κενή ερώτηση")
            continue
        if not ref:
            errors.append(f"{rid}: κενή απάντηση_αναφοράς")
        if lang == "el" and not has_greek(q):
            errors.append(f"{rid}: η ερώτηση πρέπει να είναι στα ελληνικά")
        if lang == "en" and has_greek(q):
            errors.append(f"{rid}: η ερώτηση πρέπει να είναι στα αγγλικά")
        beh = r["σωστή_συμπεριφορά"].strip()
        if beh not in BEHAVIOURS:
            errors.append(f"{rid}: σωστή_συμπεριφορά πρέπει να είναι μία από {sorted(BEHAVIOURS)}")
        if t == "near" and beh == "answer":
            errors.append(f"{rid}: ερώτηση near με συμπεριφορά «answer» — πρέπει partial ή refuse")
        if t != "near" and beh != "answer":
            warnings.append(f"{rid}: τύπος {t} με συμπεριφορά «{beh}» — σίγουρα;")
        if t == "near" and not r["τι_λείπει"].strip():
            errors.append(f"{rid}: near χωρίς «τι_λείπει»")
        key = norm(q)
        if key in seen:
            warnings.append(f"{rid}: ίδια ερώτηση με {seen[key]}")
        seen[key] = rid

        t1 = page_text(rid, r["άρθρο_1"], r["σελίδα_1"], 1) if r["σελίδα_1"].strip() or t != "near" else None
        if t1 is not None:
            check_quote(rid, r["απόσπασμα_1"], t1, 1)
            if lang == "en":
                gram = copied_ngram(q, t1)
                if gram:
                    warnings.append(f"{rid}: η ερώτηση αντιγράφει από τη σελίδα: «{gram}»")
        if t == "two_papers":
            if not r["άρθρο_2"].strip():
                errors.append(f"{rid}: two_papers χωρίς άρθρο_2")
            else:
                t2 = page_text(rid, r["άρθρο_2"], r["σελίδα_2"], 2)
                if t2 is not None:
                    check_quote(rid, r["απόσπασμα_2"], t2, 2)
                    if lang == "en":
                        gram = copied_ngram(q, t2)
                        if gram:
                            warnings.append(f"{rid}: η ερώτηση αντιγράφει από τη σελίδα 2: «{gram}»")
        if t == "table":
            items = [x for x in r["τιμές_πίνακα"].split(";") if x.strip()]
            if not items:
                errors.append(f"{rid}: table χωρίς τιμές_πίνακα")
            for it in items:
                parts = [p.strip() for p in it.split("·")]
                if len(parts) != 3 or not all(parts):
                    errors.append(f"{rid}: τιμή πίνακα «{it.strip()}» θέλει μορφή «τιμή · μονάδα · γραμμή/στήλη»")
                    continue
                if t1 is not None and norm(parts[0]) not in norm(t1):
                    errors.append(f"{rid}: η τιμή «{parts[0]}» δεν βρίσκεται στη σελίδα {r['σελίδα_1']}")

    filled = sum(bool(r["ερώτηση"].strip()) for r in rows)
    print(f"Συμπληρωμένες ερωτήσεις: {filled}/{len(rows)} · σελίδες κειμένου: {len(pages)}")
    for e in errors:
        print("ΣΦΑΛΜΑ  ", e)
    for w in warnings:
        print("ΠΡΟΣΟΧΗ ", w)
    print(f"\n{len(errors)} σφάλματα · {len(warnings)} προειδοποιήσεις")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
