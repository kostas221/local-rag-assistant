# backend/migrate_add_feedback_comment.py
"""ΜΙΑ ΦΟΡΑ: προσθέτει τη στήλη `comment` στον πίνακα feedbacks σε ΥΠΑΡΧΟΥΣΑ βάση.
Το create_all() φτιάχνει μόνο νέους πίνακες — ΔΕΝ προσθέτει στήλες σε υπάρχοντες,
οπότε χωρίς αυτό το ALTER τα INSERT με comment θα σκάνε (UndefinedColumn).
Τρέξε: docker compose exec backend python migrate_add_feedback_comment.py
"""
from sqlalchemy import text

import database


def main():
    with database.engine.begin() as conn:
        # IF NOT EXISTS -> idempotent: ακίνδυνο να ξανατρέξει (Postgres 9.6+)
        conn.execute(text(
            "ALTER TABLE feedbacks ADD COLUMN IF NOT EXISTS comment TEXT"))
    print("✅ Στήλη `comment` OK στον πίνακα feedbacks.")


if __name__ == "__main__":
    main()
