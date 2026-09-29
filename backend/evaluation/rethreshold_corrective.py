"""Κατώφλι του corrective ΕΚ ΤΩΝ ΥΣΤΕΡΩΝ, πάνω σε ένα τρέξιμο του scoreboard — ΜΗΔΕΝ κλήσεις.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (Φάση 1.1, 29/9/2026): άλλος reranker = άλλη κλίμακα. Το −3.8 του MiniLM δεν
μεταφέρεται στο gte-reranker-modernbert (βαθμοί −1.09…3.85). Αντί να μαντέψουμε και να
ξανατρέξουμε 20 λεπτά ανά δοκιμή, τρέχουμε το scoreboard ΜΙΑ φορά με CORRECTIVE_MIN_SCORE=−99
(ο corrective δέχεται ό,τι βρει) -> κάθε 2ο πέρασμα καταγράφεται με best2 ΚΑΙ σελίδες.

ΓΙΑΤΙ ΕΙΝΑΙ ΑΚΡΙΒΕΣ (όχι προσέγγιση): στο ai_core το κατώφλι χρησιμοποιείται σε ΕΝΑ σημείο,
`if sorted_final[0][0] < CORRECTIVE_MIN_SCORE: -> κομμένο` (γρ. ~1032), και best2 = ακριβώς αυτό
το sorted_final[0][0]. Άρα με αυστηρότερο κατώφλι t: ό,τι πέρασε μέσω corrective με best2 < t
γίνεται σιωπή, και ΤΙΠΟΤΑ άλλο δεν αλλάζει (ούτε άλλη ερώτηση, ούτε άλλη σελίδα).

ΚΑΝΟΝΑΣ — ΓΡΑΜΜΕΝΟΣ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (ίδια αρχή με το −3.8 του MiniLM και το −2.6 του φύλακα):
    «χωρίς απάντηση» = category out_of_corpus σε ΟΠΟΙΟΔΗΠΟΤΕ σετ (κύριο 5, συνομιλίες 2, άλλα πεδία 5).
    t = μέσο ανάμεσα στο ΥΨΗΛΟΤΕΡΟ best2 «χωρίς απάντηση» και στο αμέσως ΥΨΗΛΟΤΕΡΟ best2 ερώτησης
        ΜΕ υλικό (cov > 0). -> 0 διαρροές ooc μέσω corrective ΕΞ ΟΡΙΣΜΟΥ, μέγιστο χειρότερο περιθώριο.
    Οι κοντινές ooc ΔΕΝ είναι «χωρίς απάντηση» σε επίπεδο 1 (έχουν σχετικό υλικό εξ ορισμού — ορισμός 25/9).
    ⚠️ Βαθμονόμηση ΜΕΣΑ στο δείγμα, όπως ακριβώς και το −3.8 (n=6, χωρίς validation) -> η σύγκριση
    είναι ισότιμη, αλλά ΚΑΙ ΤΑ ΔΥΟ κατώφλια είναι αισιόδοξα.

    docker compose exec backend python evaluation/rethreshold_corrective.py \\
        evaluation/runs/scoreboard/gte_raw.json --out gte --compare evaluation/runs/scoreboard/baseline.json
    (--threshold X = χειροκίνητο κατώφλι αντί για τον κανόνα, μόνο για διερεύνηση)
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import scoreboard as SB


def rule_threshold(rows: list[dict]) -> tuple[float | None, list, list]:
    reached = [r for r in rows if r.get("best2") is not None]
    # οι κοντινές ooc έχουν κι αυτές category out_of_corpus στο σετ τους, αλλά ΔΕΝ είναι «χωρίς
    # απάντηση» σε επίπεδο 1 (ορισμός 25/9) -> εκτός. Έλεγχος: με αυτό ο κανόνας ξαναβγάζει στο
    # MiniLM (baseline.json) ακριβώς το −3.8 του Αυγούστου: μέσο των q019 −4.47 και h008 −3.16.
    no_ans = sorted(((r["best2"], r["key"]) for r in reached
                     if r["category"] == "out_of_corpus" and r["set"] != "near_ooc"),
                    reverse=True)
    with_mat = sorted((r["best2"], r["key"]) for r in reached
                      if r["category"] != "out_of_corpus" and r["cov"] > 0)
    if not no_ans:
        return None, no_ans, with_mat
    top = no_ans[0][0]
    above = [b for b, _k in with_mat if b > top]
    if not above:
        return top + 1e-6, no_ans, with_mat          # τίποτα από πάνω: κόβει όλα τα 2α περάσματα
    return (top + above[0]) / 2, no_ans, with_mat


def apply(rows: list[dict], t: float) -> tuple[list[dict], list[str]]:
    out, flipped = [], []
    for r in rows:
        r = copy.deepcopy(r)
        if r["outcome"] == "passed_corrective" and r["best2"] is not None and r["best2"] < t:
            flipped.append(r["key"])
            r.update(outcome="cut", n_pages=0, cov=0, mrr=0.0, ndcg=0.0, pages="")
            if r.get("both") is not None:
                r["both"] = 0
            if r.get("target") is not None:
                r["target"] = 0
            r["state"] = SB.state(r)
        out.append(r)
    return out, flipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="scoreboard json τρεγμένο με CORRECTIVE_MIN_SCORE=-99")
    ap.add_argument("--out", required=True, help="label του αποτελέσματος (runs/scoreboard/<out>.json)")
    ap.add_argument("--compare", default="", help="βάση σύγκρισης, π.χ. runs/scoreboard/baseline.json")
    ap.add_argument("--threshold", type=float, default=None, help="χειροκίνητο (μόνο διερεύνηση)")
    args = ap.parse_args()

    with open(args.raw, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = raw["config"]
    if cfg.get("corrective", 0) > -50:
        print(f"!! Το τρέξιμο έχει corrective {cfg.get('corrective')} — χρειάζεται −99 "
              f"(αλλιώς λείπουν τα 2α περάσματα που κόπηκαν). Σταματάω.")
        return 1

    t_rule, no_ans, with_mat = rule_threshold(raw["rows"])
    print("best2 «χωρίς απάντηση» (ooc που έφτασαν στο 2ο πέρασμα), από πάνω:")
    for b, k in no_ans:
        print(f"   {b:>7.2f}  {k}")
    print("best2 ΜΕ υλικό (cov>0), από κάτω:")
    for b, k in with_mat:
        print(f"   {b:>7.2f}  {k}")
    t = args.threshold if args.threshold is not None else t_rule
    if t is None:
        print("!! Καμία ooc δεν έφτασε στο 2ο πέρασμα — ο κανόνας δεν ορίζεται. Δώσε --threshold.")
        return 1
    how = "χειροκίνητο" if args.threshold is not None else "ΚΑΝΟΝΑΣ (γραμμένος πριν)"
    print(f"\nκατώφλι corrective: {t:.3f}  ({how})"
          + (f" · κανόνας θα έδινε {t_rule:.3f}" if args.threshold is not None and t_rule else ""))

    rows, flipped = apply(raw["rows"], t)
    print(f"πέρασαν μέσω corrective στο −99 αλλά κόβονται στο {t:.3f}: {len(flipped)} {flipped}")
    S = SB.summarize(rows, cfg["gate"])

    prev = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            prev = json.load(f)
        print(f"\nΣύγκριση με: {args.compare}  ({prev.get('label')}, {prev.get('when')})")
        print(f"  τότε: {prev.get('config')}")
    config = dict(cfg, corrective=round(t, 3), corrective_rule=how, raw=os.path.basename(args.raw))
    print(f"  τώρα: {config}")
    SB.print_table(S, prev["summary"] if prev else None)
    if prev:
        SB.print_flips(rows, prev["rows"])

    out = dict(raw, label=args.out, config=config, summary=S, rows=rows)
    jpath = os.path.join(SB.OUT_DIR, f"{args.out}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nΣώθηκε: {jpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
