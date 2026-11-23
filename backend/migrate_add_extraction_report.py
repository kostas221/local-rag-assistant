"""ΜΙΑ ΦΟΡΑ: γεμίζει την αναφορά εξαγωγής («κείμενο από Χ/Υ σελίδες») στα έγγραφα που ανέβηκαν πριν
από τις στήλες της (23/11/2026). Οι ίδιες οι στήλες προστίθενται αυτόματα στην εκκίνηση του backend
(models.ensure_document_columns)· το script τις προσθέτει κι αυτό, αν λείπουν. Συμπληρώνονται οι τιμές, διαβάζοντας ξανά κάθε PDF από τον
δίσκο. Δεν ξαναγίνεται εισαγωγή: τα chunks και η αναζήτηση μένουν ίδια. Όσα αρχεία λείπουν από τον
δίσκο μένουν κενά και αναφέρονται ονομαστικά. Ασφαλές να ξανατρέξει: πιάνει μόνο τα κενά.

Τρέξε (προσθέτει και τις στήλες, αν λείπουν· δεν χρειάζεται να έχει ξεκινήσει το backend):
    docker compose -f docker-compose.yml -f docker-compose.upgrade.yml exec backend \
        python migrate_add_extraction_report.py
"""
import os

import database
import models
import pdf_report


def main() -> int:
    added = models.ensure_document_columns(database.engine)   # δεν περιμένει το backend
    if added:
        print(f"Προστέθηκαν οι στήλες: {added}")
    db = database.SessionLocal()
    try:
        docs = db.query(models.Document).filter(models.Document.pages_total.is_(None)).all()
        filled, missing = 0, []
        for doc in docs:
            if not doc.file_path or not os.path.exists(doc.file_path):
                missing.append(doc.file_name)
                continue
            rep = pdf_report.from_pdf(doc.file_path)
            doc.pages_total = rep["pages_total"]
            doc.pages_text = rep["pages_text"]
            doc.pages_empty = pdf_report.pages_to_str(rep["pages_empty"])
            doc.pages_scanned = pdf_report.pages_to_str(rep["pages_scanned"])
            filled += 1
            print(f"  {doc.file_name}: {pdf_report.describe(doc.pages_total, doc.pages_text, rep['pages_empty'])}")
        db.commit()
        print(f"Συμπληρώθηκαν {filled}/{len(docs)}" + (f" · λείπουν από τον δίσκο: {missing}" if missing else ""))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
