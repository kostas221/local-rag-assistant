"""Δύο τρεξίματα του scoreboard: ΑΚΡΙΒΩΣ ίδια αναζήτηση; Χωρίς Docker, χωρίς μοντέλα, μόνο stdlib.

ΓΙΑΤΙ (23/11/2026): μια αλλαγή κώδικα που ΔΕΝ πρέπει να αλλάζει συμπεριφορά (π.χ. η συγχώνευση των
τριών αντιγράφων της αλυσίδας κατάταξης σε μία `_rank`) αποδεικνύεται μόνο αν, για κάθε ερώτηση, είναι
ίδια: οι σελίδες ΚΑΙ η σειρά τους, η βαθμολογία του 1ου και του 2ου περάσματος, η αναδιατύπωση, η
απόφαση του φύλακα. Το `scoreboard.py --compare` δείχνει μόνο αλλαγές κατάστασης· αυτό κοιτά όλα.

    python backend/evaluation/compare_scoreboard_runs.py g38b refactor_rank
"""
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "runs" / "scoreboard"
FIELDS = ("pages", "best1", "best2", "rewrite", "outcome", "state", "translation")


def load(label: str) -> tuple[dict, dict]:
    with open(OUT_DIR / f"{label}.json", encoding="utf-8") as f:
        data = json.load(f)
    return {r["key"]: r for r in data["rows"]}, data.get("gemini", {})


def main(old_label: str, new_label: str) -> int:
    old, old_gemini = load(old_label)
    new, gemini = load(new_label)
    common = sorted(old.keys() & new.keys())
    diffs = {f: [] for f in FIELDS}
    for k in common:
        for f in FIELDS:
            if old[k].get(f) != new[k].get(f):
                diffs[f].append(k)
    print(f"{old_label} -> {new_label}: κοινές ερωτήσεις {len(common)} "
          f"(μόνο στο παλιό {len(old.keys() - new.keys())}, μόνο στο νέο {len(new.keys() - old.keys())})")
    print(f"Gemini, νέες κλήσεις: παλιό {old_gemini.get('new')} · νέο {gemini.get('new')} "
          f"(πρέπει 0)\nΣφάλματα παλιού: {old_gemini.get('errors')}\nΣφάλματα νέου:  {gemini.get('errors')}")
    for f in FIELDS:
        n_same = len(common) - len(diffs[f])
        print(f"  {f:<12} ίδια {n_same}/{len(common)}" + (f"   διαφέρουν: {diffs[f][:10]}" if diffs[f] else ""))
    changed = sorted({k for f in FIELDS for k in diffs[f]})
    for k in changed[:5]:
        print(f"\n  {k}")
        for f in FIELDS:
            if old[k].get(f) != new[k].get(f):
                print(f"    {f}: {old[k].get(f)!r}\n    {'':<{len(f)}}  -> {new[k].get(f)!r}")
    # Σφάλματα «λείπει από το πάγωμα» επιτρέπονται: αφορούν κλήσεις που απέτυχαν και στο παλιό
    # τρέξιμο, άρα δεν αποθηκεύτηκαν ποτέ. Αν άλλαζε κάτι εξαιτίας τους, θα φαινόταν στα πεδία.
    ok = not changed and not (old.keys() ^ new.keys()) and not gemini.get("new")
    print("\nΙΔΙΑ ΑΝΑΖΗΤΗΣΗ ✓" if ok else "\nΔΙΑΦΟΡΕΣ ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
