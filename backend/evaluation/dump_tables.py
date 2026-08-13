"""Βρίσκει ΚΑΘΕ πίνακα του σώματος και τον τυπώνει ΟΠΩΣ ΤΟΝ ΒΛΕΠΕΙ ΤΟ ΜΟΝΤΕΛΟ.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ
-------------
Το `q027` έδειξε ότι η εξαγωγή ισοπεδώνει τους πίνακες κατά στήλη και το μοντέλο
μέτρησε 5 γραμμές αντί για 6. Το σύνολο αξιολόγησης όμως έχει **μία** ερώτηση
πίνακα σε 77, ενώ το σώμα έχει **είκοσι** πίνακες. Άρα δεν ξέρουμε αν το q027
είναι εξαίρεση ή κανόνας.

Για να γραφτούν ερωτήσεις πινάκων με ΑΚΡΙΒΕΙΣ αναμενόμενες τιμές χρειάζεται να
φανεί πρώτα τι ακριβώς φτάνει στο prompt. Δεν διαβάζουμε το PDF με τα μάτια —
διαβάζουμε το κείμενο ΜΕΤΑ την εξαγωγή, γιατί αυτό βλέπει το Gemini.

ΜΗΔΕΝ μοντέλα, ΜΗΔΕΝ κλήσεις API, ~5 δευτερόλεπτα.

    docker compose exec backend python evaluation/dump_tables.py
    docker compose exec backend python evaluation/dump_tables.py --doc 1902 --context 1800
"""
import argparse
import os
import re
import sys

import fitz

DOCS = "/app/uploaded_docs"
CAPTION = re.compile(r"^[ \t]*Table\s+(\d+)\s*[:.]", re.M)


def real_name(doc) -> str:
    """Τα αρχεία είναι αποθηκευμένα με uuid· το πραγματικό όνομα είναι στα metadata."""
    for key in ("title", "subject"):
        v = (doc.metadata or {}).get(key) or ""
        if v.strip():
            return v.strip()[:60]
    return ""


def main(args) -> int:
    files = sorted(f for f in os.listdir(DOCS) if f.lower().endswith(".pdf"))
    total = 0
    for fname in files:
        path = os.path.join(DOCS, fname)
        doc = fitz.open(path)
        label = real_name(doc) or fname
        for pno in range(doc.page_count):
            text = doc[pno].get_text()
            caps = CAPTION.findall(text)
            if not caps:
                continue
            if args.doc and args.doc.lower() not in (fname + label).lower():
                continue
            total += len(caps)
            print("=" * 78)
            print(f"{label}  ·  αρχείο {fname[:16]}  ·  σελίδα {pno}  ·  "
                  f"Πίνακας {', '.join(caps)}")
            print("=" * 78)
            body = text.strip()
            if len(body) > args.context:
                # Κρατάμε γύρω από την πρώτη λεζάντα, εκεί είναι ο πίνακας.
                m = CAPTION.search(text)
                start = max(0, m.start() - args.context)
                body = text[start:m.start() + 600].strip()
                print("[... κομμένο ...]")
            print(body)
            print()
        doc.close()
    print("=" * 78)
    print(f"ΣΥΝΟΛΟ λεζαντών: {total} σε {len(files)} έγγραφα")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Οι πίνακες όπως τους βλέπει το μοντέλο.")
    ap.add_argument("--doc", default=None, help="φίλτρο ονόματος αρχείου/τίτλου")
    ap.add_argument("--context", type=int, default=2400,
                    help="χαρακτήρες πριν τη λεζάντα")
    sys.exit(main(ap.parse_args()))
