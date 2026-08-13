"""ΜΙΑ ΦΟΡΑ: προσθέτει τη στήλη `file_hash` στον πίνακα documents σε ΥΠΑΡΧΟΥΣΑ βάση.
Το create_all() φτιάχνει μόνο νέους πίνακες — ΔΕΝ προσθέτει στήλες σε υπάρχοντες,
οπότε χωρίς αυτό το ALTER κάθε query πάνω στο file_hash θα σκάει (UndefinedColumn).

Γεμίζει ΚΑΙ τα ήδη ανεβασμένα έγγραφα: χωρίς αυτό ο έλεγχος διπλότυπου δεν θα
έπιανε ακριβώς τα αρχεία που είναι πιθανότερο να ξαναανέβουν κατά λάθος.
Όσα αρχεία λείπουν από τον δίσκο μένουν NULL και αναφέρονται ονομαστικά.

Τρέξε: docker compose exec backend python migrate_add_file_hash.py
"""
import hashlib
import os

from sqlalchemy import text

import database


def sha256_of(path: str) -> str:
    """Ίδιο μοτίβο με το /upload: 1MB τη φορά, ώστε ένα 50MB PDF να μη μπει
    ολόκληρο στη μνήμη."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    with database.engine.begin() as conn:
        # IF NOT EXISTS -> idempotent: ακίνδυνο να ξανατρέξει (Postgres 9.6+)
        conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_documents_file_hash "
            "ON documents (file_hash)"))

        rows = conn.execute(text(
            "SELECT id, file_path, file_name FROM documents "
            "WHERE file_hash IS NULL")).fetchall()

        filled, missing = 0, []
        for doc_id, file_path, file_name in rows:
            if not file_path or not os.path.exists(file_path):
                missing.append(f"{doc_id}:{file_name}")
                continue
            conn.execute(
                text("UPDATE documents SET file_hash = :h WHERE id = :i"),
                {"h": sha256_of(file_path), "i": doc_id})
            filled += 1

    print(f"✅ Στήλη `file_hash` OK στον πίνακα documents "
          f"({filled} έγγραφα ενημερώθηκαν).")
    if missing:
        print(f"⚠️  Χωρίς αρχείο στον δίσκο, έμειναν NULL: {', '.join(missing)}")


if __name__ == "__main__":
    main()
