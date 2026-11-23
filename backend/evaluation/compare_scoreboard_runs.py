"""Δύο τρεξίματα του scoreboard: ΑΚΡΙΒΩΣ ίδια αναζήτηση; Χωρίς Docker, χωρίς μοντέλα, μόνο stdlib.

ΓΙΑΤΙ (23/11/2026): μια αλλαγή κώδικα που ΔΕΝ πρέπει να αλλάζει συμπεριφορά (π.χ. η συγχώνευση των
τριών αντιγράφων της αλυσίδας κατάταξης σε μία `_rank`) αποδεικνύεται μόνο αν, για κάθε ερώτηση, είναι
ίδια: οι σελίδες ΚΑΙ η σειρά τους, η βαθμολογία του 1ου και του 2ου περάσματος, η αναδιατύπωση, η
απόφαση του φύλακα. Το `scoreboard.py --compare` δείχνει μόνο αλλαγές κατάστασης· αυτό κοιτά όλα.

    python backend/evaluation/compare_scoreboard_runs.py g38b refactor_rank

--only-outcome passed_corrective (βήμα 2β): η αλλαγή ΠΡΕΠΕΙ να αγγίζει μόνο τις ερωτήσεις που στο παλιό
τρέξιμο απαντήθηκαν από τη δεύτερη αναζήτηση. Αλλαγή σε οποιαδήποτε άλλη ερώτηση = λάθος. Νέες κλήσεις
επιτρέπονται εδώ (είναι οι κλήσεις του κριτή), αλλά μετρώνται.

    python backend/evaluation/compare_scoreboard_runs.py g38b veto --only-outcome passed_corrective
"""
import argparse
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "runs" / "scoreboard"
FIELDS = ("pages", "best1", "best2", "rewrite", "outcome", "state", "translation")


def load(label: str) -> tuple[dict, dict]:
    with open(OUT_DIR / f"{label}.json", encoding="utf-8") as f:
        data = json.load(f)
    return {r["key"]: r for r in data["rows"]}, data.get("gemini", {})


def main(old_label: str, new_label: str, only_outcome: str | None = None) -> int:
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
    print(f"Gemini, νέες κλήσεις: παλιό {old_gemini.get('new')} · νέο {gemini.get('new')}"
          f"{'' if only_outcome else ' (πρέπει 0)'}\nΣφάλματα παλιού: {old_gemini.get('errors')}\n"
          f"Σφάλματα νέου:  {gemini.get('errors')}")
    for f in FIELDS:
        n_same = len(common) - len(diffs[f])
        print(f"  {f:<12} ίδια {n_same}/{len(common)}" + (f"   διαφέρουν: {diffs[f][:10]}" if diffs[f] else ""))
    changed = sorted({k for f in FIELDS for k in diffs[f]})

    if only_outcome:
        allowed = [k for k in common if old[k].get("outcome") == only_outcome]
        outside = [k for k in changed if k not in allowed]
        print(f"\nΕρωτήσεις με «{only_outcome}» στο {old_label}: {len(allowed)} · άλλαξαν {len(changed)} "
              f"· ΕΚΤΟΣ αυτών {len(outside)}")
        for k in allowed:
            mark = "ΑΛΛΑΞΕ" if k in changed else "ίδια  "
            print(f"  {mark} {k:<16} {old[k].get('state', '')[:28]:<28} -> {new[k].get('state', '')[:28]}")
        for k in outside[:10]:
            print(f"  !! αλλαγή ΕΚΤΟΣ: {k}  {old[k].get('state')} -> {new[k].get('state')}")
        ok = not outside and not (old.keys() ^ new.keys())
        print("\nΑΛΛΑΞΑΝ ΜΟΝΟ ΟΙ ΑΝΑΜΕΝΟΜΕΝΕΣ ✓" if ok else "\nΑΛΛΑΓΕΣ ΕΚΤΟΣ ΤΩΝ ΑΝΑΜΕΝΟΜΕΝΩΝ ✗")
        return 0 if ok else 1

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--only-outcome", default=None,
                    help="επιτρέπονται αλλαγές ΜΟΝΟ σε ερωτήσεις με αυτό το outcome στο παλιό τρέξιμο")
    a = ap.parse_args()
    sys.exit(main(a.old, a.new, a.only_outcome))
