# backend/migrate_add_conversation_created_at.py
"""ΜΙΑ ΦΟΡΑ: προσθέτει τη στήλη `created_at` στον πίνακα conversations.

Το create_all() φτιάχνει μόνο ΝΕΟΥΣ πίνακες — δεν προσθέτει στήλες σε
υπάρχοντες, οπότε χωρίς αυτό το ALTER κάθε SELECT θα σκάει (UndefinedColumn)
μόλις το μοντέλο μάθει τη στήλη.

ΓΙΑΤΙ ΔΥΟ ΞΕΧΩΡΙΣΤΑ ΒΗΜΑΤΑ ΚΑΙ ΟΧΙ `ADD COLUMN ... DEFAULT now()`:
Ένα ADD COLUMN με DEFAULT γεμίζει ΚΑΙ τις υπάρχουσες γραμμές με την ώρα του
migration. Θα ήταν ψέμα: αυτές οι συνομιλίες φτιάχτηκαν κάποτε άλλοτε και η
ώρα τους δεν είναι ανακτήσιμη. Οπότε πρώτα η στήλη ΧΩΡΙΣ default (παλιές
γραμμές -> NULL, δηλαδή «άγνωστο») και μετά το default για τις ΕΠΟΜΕΝΕΣ.

Τρέξε: docker compose exec backend python migrate_add_conversation_created_at.py
"""
from sqlalchemy import text

import database


def main():
    with database.engine.begin() as conn:
        # 1) Η στήλη, ΧΩΡΙΣ default -> οι υπάρχουσες γραμμές μένουν NULL.
        #    IF NOT EXISTS -> idempotent, ακίνδυνο να ξανατρέξει (Postgres 9.6+).
        conn.execute(text(
            "ALTER TABLE conversations "
            "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ"))

        # 2) Το default ΜΟΝΟ για τις επόμενες εγγραφές. Αντιστοιχεί στο
        #    server_default=func.now() του μοντέλου: η ώρα μπαίνει από τη βάση.
        conn.execute(text(
            "ALTER TABLE conversations "
            "ALTER COLUMN created_at SET DEFAULT now()"))

        total = conn.execute(text(
            "SELECT COUNT(*) FROM conversations")).scalar()
        unknown = conn.execute(text(
            "SELECT COUNT(*) FROM conversations "
            "WHERE created_at IS NULL")).scalar()

    print("OK: στήλη `created_at` στον πίνακα conversations.")
    print("   συνομιλίες           %d" % total)
    print("   με άγνωστη ώρα       %d  (NULL — προϋπήρχαν της στήλης)" % unknown)
    print("   νέες συνομιλίες παίρνουν πλέον ώρα από τη βάση (now()).")


if __name__ == "__main__":
    main()
