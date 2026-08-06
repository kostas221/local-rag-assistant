"""Σβήνει τις αποθηκευμένες μεταφράσεις ώστε να ξαναπαραχθούν με το νέο prompt.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το _translation_cache έχει key την ΕΡΩΤΗΣΗ, όχι το prompt. Άρα κάθε αλλαγή στο
optimize_query() είναι ΑΟΡΑΤΗ για ερωτήσεις που έχουν ήδη μεταφραστεί — η μέτρηση
θα έδειχνε «καμία διαφορά» και θα βγάζαμε λάθος συμπέρασμα. Αυτό ακριβώς το είδος
σιωπηλής μόλυνσης έχει ήδη χαλάσει μετρήσεις σε αυτό το project.

ΤΙ ΚΑΝΕΙ: κρατάει backup με timestamp, μετά σβήνει τα επιλεγμένα keys.
Το cache ξαναχτίζεται μόνο του στην επόμενη ερώτηση (1 κλήση Gemini ανά ερώτηση).

    docker compose exec backend python evaluation/reset_translation_cache.py --dry-run
    docker compose exec backend python evaluation/reset_translation_cache.py
    docker compose exec backend python evaluation/reset_translation_cache.py --all

ΜΕΤΑ ΤΟ ΣΒΗΣΙΜΟ: docker compose restart backend
(το cache είναι φορτωμένο στη μνήμη της διεργασίας — αλλιώς η παλιά έκδοση μένει
ενεργή και ξαναγράφεται στο αρχείο.)
"""
import argparse
import json
import os
import shutil
import sys
import time

CACHE = "/app/vector_db/_translation_cache.json"
GREEK = set(range(0x370, 0x400)) | set(range(0x1F00, 0x2000))


def has_greek(s: str) -> bool:
    return any(ord(c) in GREEK for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="σβήσε τα πάντα, όχι μόνο τα ελληνικά keys")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", default=CACHE)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"Δεν υπάρχει: {args.path}")
        return 1

    with open(args.path, encoding="utf-8") as f:
        cache = json.load(f)

    victims = [k for k in cache if args.all or has_greek(k)]
    print(f"Cache: {len(cache)} entries · προς διαγραφή: {len(victims)}\n")
    for k in victims:
        print(f"  {k[:70]}...")
        print(f"     -> {cache[k][:70]}")

    if args.dry_run:
        print("\n--dry-run: δεν έγινε καμία αλλαγή.")
        return 0
    if not victims:
        print("Τίποτα προς διαγραφή.")
        return 0

    bak = f"{args.path}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.path, bak)
    for k in victims:
        del cache[k]
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\nbackup:  {bak}")
    print(f"έμειναν: {len(cache)} entries")
    print("\nΤΩΡΑ: docker compose restart backend  (το cache είναι in-memory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
