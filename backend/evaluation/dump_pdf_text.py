"""Εξάγει το κείμενο ενός PDF ανά σελίδα σε .txt — ΜΟΝΟ για να διαβάσουμε το
περιεχόμενο και να γράψουμε ερωτήσεις. Ίδιος extractor με το production
(pymupdf + NFKC), ώστε αυτό που βλέπουμε εδώ να είναι αυτό που θα μπει στα chunks.

    docker compose exec backend python evaluation/dump_pdf_text.py evaluation/test_papers/foo.pdf
"""
import sys
import unicodedata

import fitz  # pymupdf


def main():
    path = sys.argv[1]
    out = path.rsplit(".", 1)[0] + ".txt"
    doc = fitz.open(path)
    with open(out, "w", encoding="utf-8") as f:
        for i, page in enumerate(doc, start=1):
            text = unicodedata.normalize("NFKC", page.get_text())
            f.write(f"\n===== PAGE {i}/{len(doc)} =====\n")
            f.write(text)
    print(f"Γράφτηκε: {out}  ({len(doc)} σελίδες)")


if __name__ == "__main__":
    main()
