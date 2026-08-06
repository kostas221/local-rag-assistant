"""Φτιάχνει υποσύνολο του golden set για judge run περιορισμένου κόστους,
και συγκρίνει το αποτέλεσμα ΑΝΑ ΕΡΩΤΗΣΗ με προηγούμενο judge run.

ΓΙΑΤΙ ΥΠΟΣΥΝΟΛΟ: ένα πλήρες judge run είναι ~100 κλήσεις Gemini. Για να
ελέγξουμε αν το THINKING_BUDGET=0 υποβάθμισε τις απαντήσεις, αρκεί δείγμα από
τις κατηγορίες όπου η σκέψη υποτίθεται ότι βοηθάει περισσότερο.

ΓΙΑΤΙ ΑΝΑ ΕΡΩΤΗΣΗ: οι μέσοι όροι κρύβουν αντισταθμίσεις — μία ερώτηση που πέφτει
από 5 σε 3 και μία που ανεβαίνει από 4 σε 5 δίνουν ίδιο μέσο όρο με το «τίποτα
δεν άλλαξε». Με ids συγκρίνουμε την ΙΔΙΑ ερώτηση με τον εαυτό της.

    # 1. φτιάξε το υποσύνολο
    docker compose exec backend python evaluation/make_judge_subset.py build
    # 2. τρέξε το judge run πάνω του (καίει quota)
    docker compose exec backend python run_eval.py evaluation/golden_subset_15.jsonl \
        --out evaluation/judge_thinking0.csv
    # 3. σύγκρινε με το baseline
    docker compose exec backend python evaluation/make_judge_subset.py compare \
        evaluation/judge_thinking0.csv
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_set_50.jsonl")
SUBSET = os.path.join(HERE, "golden_subset_15.jsonl")
BASELINE = os.path.join(HERE, "judge_v2.csv")

# Οι κατηγορίες όπου το thinking θα έπρεπε να μετράει, και πόσες από κάθε μία.
# direct_fact εξαιρείται σκόπιμα: είναι απλή εξαγωγή γεγονότος, το πιο αδιάφορο
# τεστ για τη σκέψη. out_of_corpus μπαίνει με 1 ώστε να ελεγχθεί ότι η άρνηση
# (το gate + το system prompt) εξακολουθεί να δουλεύει.
QUOTAS = {"enumeration": 5, "reasoning": 5, "multi_hop": 4, "out_of_corpus": 1}
METRICS = ["accuracy", "completeness", "relevance", "faithfulness"]


def build() -> int:
    picked, counts = [], dict.fromkeys(QUOTAS, 0)
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            cat = t.get("category")
            if cat in QUOTAS and counts[cat] < QUOTAS[cat]:
                picked.append(t)
                counts[cat] += 1
    with open(SUBSET, "w", encoding="utf-8") as f:
        for t in picked:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Γράφτηκε {SUBSET}: {len(picked)} ερωτήσεις")
    for cat, n in counts.items():
        print(f"  {cat:<16} {n}")
    print(f"\nΚόστος judge run: ~{2 * len(picked)} κλήσεις Gemini.")
    return 0


def read_csv(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def compare(new_path: str) -> int:
    base, new = read_csv(BASELINE), read_csv(new_path)
    shared = [i for i in new if i in base]
    if not shared:
        print("Καμία κοινή ερώτηση — λάθος αρχεία;")
        return 1

    print(f"Κοινές ερωτήσεις: {len(shared)}\n")
    print(f"{'id':<7}{'κατηγορία':<15}" + "".join(f"{m[:4]:>13}" for m in METRICS))
    print("-" * 74)
    deltas = {m: [] for m in METRICS}
    regressions = []
    for qid in sorted(shared):
        b, n = base[qid], new[qid]
        cells = []
        for m in METRICS:
            try:
                bv, nv = float(b[m]), float(n[m])
            except (ValueError, KeyError, TypeError):
                cells.append(f"{'—':>13}")
                continue
            d = nv - bv
            deltas[m].append(d)
            if d <= -0.5:
                regressions.append((qid, m, bv, nv))
            flag = "" if abs(d) < 0.01 else ("+" if d > 0 else "")
            cells.append(f"{bv:>5.1f}->{nv:<4.1f}{flag:>2}"
                         if abs(d) >= 0.01 else f"{nv:>13.1f}")
        print(f"{qid:<7}{n.get('category', '')[:14]:<15}" + "".join(cells))

    print("\n--- ΜΕΣΕΣ ΜΕΤΑΒΟΛΕΣ ---")
    for m in METRICS:
        ds = deltas[m]
        if ds:
            print(f"  {m:<14} {sum(ds) / len(ds):+.3f}")

    if regressions:
        print(f"\n!!! ΥΠΟΒΑΘΜΙΣΕΙΣ >= 0.5 ({len(regressions)}):")
        for qid, m, bv, nv in regressions:
            print(f"  {qid} {m}: {bv:.1f} -> {nv:.1f}")
    else:
        print("\nΚαμία υποβάθμιση >= 0.5 σε καμία ερώτηση/μετρική.")

    # ΠΡΟΣΟΧΗ ΣΤΗΝ ΕΡΜΗΝΕΙΑ: ο judge είναι Gemini, δηλαδή ΜΗ ντετερμινιστικός —
    # η ίδια ερώτηση παίρνει άλλοτε 4 κι άλλοτε 5. Μεμονωμένη υποβάθμιση ΔΕΝ
    # αποδεικνύει αιτιότητα. Αυτό που αποδεικνύει είναι το ΜΟΤΙΒΟ: υποβαθμίσεις
    # συγκεντρωμένες σε μία κατηγορία, που εξαφανίζονται όταν αντιστραφεί η
    # αλλαγή. Γι' αυτό συγκρίνεται πάντα ΖΕΥΓΟΣ συνθηκών, όχι μία.
    print("\nΕΡΜΗΝΕΙΑ: μία μεμονωμένη μεταβολή ±1 είναι μέσα στον θόρυβο του "
          "judge. Συμπέρασμα βγάζει μόνο το μοτίβο σε δύο runs — υποβάθμιση "
          "που εμφανίζεται στη ΜΙΑ συνθήκη και φεύγει στην ΑΛΛΗ.")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "compare"):
        print(__doc__)
        return 1
    if sys.argv[1] == "build":
        return build()
    if len(sys.argv) < 3:
        print("Λείπει η διαδρομή του νέου CSV.")
        return 1
    return compare(sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
