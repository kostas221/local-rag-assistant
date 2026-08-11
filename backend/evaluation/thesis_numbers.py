"""Τυπώνει ΚΑΘΕ νούμερο που μπαίνει στη διπλωματική, υπολογισμένο επιτόπου.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ
-------------
Τα νούμερα ζουν σε ~12 διαφορετικά CSV. Κάθε φορά που αντιγράφονται με το χέρι
σε πίνακα υπάρχει ρίσκο μεταγραφής — και ΕΧΕΙ ΣΥΜΒΕΙ: το «750 scaled MRR −0.077»
και το «1000 scaled cov +7.3pp» γράφτηκαν λάθος στο CLAUDE.md και βρέθηκαν μόνο
όταν ξανατρέξαμε το πείραμα. Εδώ κάθε νούμερο υπολογίζεται από το artifact του,
οπότε δεν υπάρχει βήμα αντιγραφής όπου να χαθεί κάτι.

Όπου ΔΕΝ υπάρχει artifact (latency, concurrency, κόστος, scaling) τυπώνεται η
εντολή που το παράγει αντί για καρφωμένο νούμερο. Καλύτερα να λείπει ένα νούμερο
παρά να υπάρχει ένα που κανείς δεν μπορεί να ελέγξει.

ΜΗΔΕΝ εξαρτήσεις, ΜΗΔΕΝ μοντέλα, ΜΗΔΕΝ κλήσεις API. Τρέχει σε ~1 s.

    docker compose exec backend python evaluation/thesis_numbers.py
    docker compose exec backend python evaluation/thesis_numbers.py --section 5.5
"""
import argparse
import contextlib
import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# Το CSV του κύριου συνόλου είναι το ΠΡΟΕΠΙΛΕΓΜΕΝΟ αρχείο αποτελεσμάτων· έχει ήδη
# γραφτεί από πάνω μία φορά από τρέξιμο άλλου συνόλου, γι' αυτό ελέγχεται το n.
MAIN = os.path.join(HERE, "results_retrieval.csv")

CATS = ["direct_fact", "enumeration", "reasoning", "multi_hop", "out_of_corpus"]


def read(path):
    """CSV -> λίστα από dicts. utf-8-sig: τα αρχεία γράφτηκαν σε Windows (BOM)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def num(rows, key):
    """Οι στήλες βαθμολογίας είναι κενές όπου ο κριτής δεν έτρεξε."""
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if v:
            with contextlib.suppress(ValueError):
                out.append(float(v))
    return out


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def title(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def missing(name, cmd):
    print(f"  [ΧΩΡΙΣ ARTIFACT] {name}")
    print(f"      -> {cmd}")


# ---------------------------------------------------------------- 5.2 σύνολα
def sec_sets():
    title("5.2  ΣΥΝΟΛΑ ΑΞΙΟΛΟΓΗΣΗΣ")
    for fn in ["golden_set_50", "golden_multihop_new",
               "golden_hard_paraphrase", "golden_conversations"]:
        p = os.path.join(HERE, fn + ".jsonl")
        if not os.path.exists(p):
            print(f"  {fn:26s} ΛΕΙΠΕΙ")
            continue
        with open(p, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        by = {}
        for r in rows:
            by[r.get("category", "?")] = by.get(r.get("category", "?"), 0) + 1
        detail = " · ".join(f"{k} {v}" for k, v in sorted(by.items()))
        print(f"  {fn:26s} n={len(rows):<4d} {detail}")


# ------------------------------------------------------------ 5.4 ανάκτηση
def sec_retrieval():
    title("5.4  ΑΝΑΚΤΗΣΗ  (πηγή: results_retrieval.csv)")
    rows = read(MAIN)
    if not rows:
        missing("ανάκτηση", "run_eval.py evaluation/golden_set_50.jsonl --retrieval-only")
        return
    if len(rows) != 50:
        print(f"  ⚠ ΠΡΟΣΟΧΗ: {len(rows)} γραμμές αντί για 50 — λάθος σύνολο στο αρχείο!")
    inc = [r for r in rows if r["category"] != "out_of_corpus"]
    print(f"  in-corpus n={len(inc)}")
    print(f"     MRR {mean(num(inc, 'mrr')):.3f} · nDCG {mean(num(inc, 'ndcg')):.3f}"
          f" · κάλυψη {mean(num(inc, 'keyword_coverage')):.1f}%")
    print("  ανά κατηγορία:")
    for c in CATS:
        sub = [r for r in rows if r["category"] == c]
        if sub:
            print(f"     {c:<15s} n={len(sub):<3d} MRR {mean(num(sub, 'mrr')):.3f}"
                  f" · nDCG {mean(num(sub, 'ndcg')):.3f}"
                  f" · κάλυψη {mean(num(sub, 'keyword_coverage')):.1f}%")


def sec_ladder():
    title("5.4  ΘΕΤΙΚΗ ΣΚΑΛΑ  (πηγή: runs/ladder.csv)")
    rows = read(os.path.join(RUNS, "ladder.csv"))
    if not rows:
        missing("ablation ladder", "evaluation/ablation_ladder.py")
        return
    inc = [r for r in rows if r["category"] != "out_of_corpus"]
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]
    stages = [("dense μόνο", "dense"), ("+ BM25 / RRF", "+bm25/rrf"),
              ("+ cross-encoder", "+reranker"), ("+ relevance gate", "+gate")]
    prev = None
    for label, key in stages:
        m = mean(num(inc, key + "_mrr"))
        blocked = sum(1 for r in ooc
                      if (r.get(key + "_blocked") or "").strip().lower()
                      in ("true", "1", "yes"))
        delta = "" if prev is None else f"  (Δ {m - prev:+.3f})"
        print(f"  {label:<18s} MRR {m:.3f}{delta:<14s} ooc σωστά σιωπηλά {blocked}/{len(ooc)}")
        prev = m


def sec_gate():
    title("5.4  ΚΑΤΩΦΛΙ ΣΥΝΑΦΕΙΑΣ  (πηγή: runs/gate_margin_l12.csv)")
    rows = read(os.path.join(RUNS, "gate_margin_l12.csv"))
    if not rows:
        missing("gate margin", "evaluation/measure_gate_margin.py --csv ...")
        return
    inc = sorted(num([r for r in rows if r["category"] != "out_of_corpus"], "best"))
    ooc = sorted(num([r for r in rows if r["category"] == "out_of_corpus"], "best"))
    # ΙΔΙΟΣ τύπος με το measure_gate_margin.py που παρήγαγε το CSV — αλλιώς το p10
    # βγαίνει άλλο νούμερο από το καταγεγραμμένο για τον ίδιο ακριβώς πίνακα.
    p10 = inc[max(0, len(inc) // 10)]
    print(f"  in-corpus (n={len(inc)}):  min {inc[0]:.2f} · p10 {p10:.2f}"
          f" · διάμεσος {statistics.median(inc):.2f} · max {inc[-1]:.2f}")
    print(f"  out_of_corpus (n={len(ooc)}):  max {ooc[-1]:.2f} · min {ooc[0]:.2f}")
    print(f"  ΚΕΝΟ (min in-corpus − max ooc) = {inc[0] - ooc[-1]:.2f}")
    print(f"  ΟΛΟ το εύρος [{ooc[-1]:.2f}, {inc[0]:.2f}] δίνει {len(inc) + len(ooc)}"
          f"/{len(inc) + len(ooc)} σωστά")


# ---------------------------------------------------------------- 5.3 chunk
def sec_chunk():
    title("5.3  ΜΕΓΕΘΟΣ ΚΟΜΜΑΤΙΟΥ  (πηγή: runs/chunk_*.csv)")
    found = False
    print(f"  {'σύνολο':<9s}{'βάθη':<9s}{'chunk':>7s}{'chunks':>9s}"
          f"{'MRR':>8s}{'κάλυψη':>10s}{'s/ερώτ':>9s}")
    for name, label in [("chunk_main_scaled", "κύριο"), ("chunk_main_fixed", "κύριο"),
                        ("chunk_hard_scaled", "δύσκολο"), ("chunk_hard_fixed", "δύσκολο")]:
        rows = read(os.path.join(RUNS, name + ".csv"))
        if not rows:
            continue
        found = True
        for r in rows:
            print(f"  {label:<9s}{r['depth_mode']:<9s}{r['chunk_size']:>7s}"
                  f"{int(r['chunks']):>9,d}{float(r['mrr']):>8.3f}"
                  f"{float(r['coverage']):>9.2f}%{float(r['s_per_query']):>9.2f}")
    if not found:
        missing("chunk sweep",
                "chunk_experiment.py --sizes 750,1000,1500 --depth-mode scaled --csv ...")
    else:
        print("  ΣΗΜ: στα 1500 οι δύο μέθοδοι βαθών ΕΙΝΑΙ η ίδια συνθήκη (συντελεστής 1.0).")
        print("       Διαφορά εκεί = θόρυβος. Στο δύσκολο σύνολο ο corrective agent")
        print("       τρέχει και ΔΕΝ είναι ντετερμινιστικός χωρίς ENABLE_CORRECTIVE=0.")


def sec_candidates():
    title("5.3  ΒΑΘΟΣ ΥΠΟΨΗΦΙΩΝ  N=10 vs N=15  (πηγή: runs/r_n1*.csv)")
    pairs = [("κύριο", os.path.join(RUNS, "r_n10_main.csv"), MAIN),
             ("δύσκολο", os.path.join(RUNS, "r_n10_hard.csv"),
              os.path.join(RUNS, "r_n15_hard.csv"))]
    for label, p10, p15 in pairs:
        a, b = read(p10), read(p15)
        if not a or not b:
            missing(f"N=10/N=15 ({label})",
                    "run_eval.py ... --retrieval-only --out runs/r_n10_*.csv")
            continue
        ka = {r["id"]: r for r in a}
        kb = {r["id"]: r for r in b}
        inc_a = [r for r in a if r["category"] != "out_of_corpus"]
        inc_b = [r for r in b if r["category"] != "out_of_corpus"]
        print(f"  {label}:  N=10 MRR {mean(num(inc_a, 'mrr')):.3f}"
              f" κάλυψη {mean(num(inc_a, 'keyword_coverage')):.1f}%"
              f"   |   N=15 MRR {mean(num(inc_b, 'mrr')):.3f}"
              f" κάλυψη {mean(num(inc_b, 'keyword_coverage')):.1f}%")
        diff = [k for k in ka if k in kb
                and (ka[k]["mrr"], ka[k]["keyword_coverage"])
                != (kb[k]["mrr"], kb[k]["keyword_coverage"])]
        print(f"     ταυτόσημες {len(ka) - len(diff)}/{len(ka)} · διαφορετικές {len(diff)}")
        for k in sorted(diff):
            c10, c15 = float(ka[k]["keyword_coverage"]), float(kb[k]["keyword_coverage"])
            m10, m15 = float(ka[k]["mrr"]), float(kb[k]["mrr"])
            who = "N=15" if (c15, m15) > (c10, m10) else "N=10"
            print(f"       {k:<6s} κάλυψη {c10:5.1f} -> {c15:5.1f}"
                  f"   MRR {m10:.3f} -> {m15:.3f}   υπέρ {who}")


# ---------------------------------------------------------------- 5.5 judge
CRIT = ["accuracy", "completeness", "relevance", "faithfulness"]


def _judge_block(rows):
    scored = [r for r in rows if (r.get("accuracy") or "").strip()]
    nomh = [r for r in scored if r["category"] != "multi_hop"]
    for label, sub in [(f"όλες ({len(scored)})", scored),
                       (f"χωρίς πολλαπλών εγγράφων ({len(nomh)})", nomh)]:
        vals = "  ".join(f"{c[:4]} {mean(num(sub, c)):.2f}" for c in CRIT)
        print(f"    {label:<34s} {vals}")
    perfect = sum(1 for r in scored
                  if all((r.get(c) or "").strip() in ("5", "5.0") for c in CRIT))
    print(f"    τέλειες 5/5/5/5: {perfect}/{len(scored)}")
    return scored


def sec_judge():
    title("5.5  ΠΟΙΟΤΗΤΑ ΑΠΑΝΤΗΣΕΩΝ")
    # ⚠ ΠΡΟΕΛΕΥΣΗ: ο ΠΛΗΡΗΣ κριτής (n=50) έτρεξε στο MiniLM-L-6 (commit 6d0e598,
    # 7/8) και ΔΕΝ ξανατρέχτηκε μετά την αναβάθμιση σε L-12 — σκόπιμα, γιατί με
    # 48/50 στο ταβάνι ένα δεύτερο τρέξιμο δεν μπορεί να δείξει βελτίωση. Ο έλεγχος
    # του L-12 έγινε στο υποσύνολο ΡΙΣΚΟΥ των 12. Το κείμενο πρέπει να το λέει.
    full = read(os.path.join(HERE, "judge_minilm.csv"))
    if not full:
        missing("πλήρης κριτής", "run_eval.py evaluation/golden_set_50.jsonl (ΚΑΙΕΙ QUOTA)")
    else:
        print("  ΠΛΗΡΕΣ ΣΥΝΟΛΟ — judge_minilm.csv  [σύστημα MiniLM-L-6, ΔΕΝ ξανατρέχτηκε]")
        _judge_block(full)

    sub = read(os.path.join(RUNS, "judge_l12.csv"))
    if not sub:
        missing("υποσύνολο ρίσκου L-12", "evaluation/make_risk_subset.py --n 12")
    else:
        print("\n  ΥΠΟΣΥΝΟΛΟ ΡΙΣΚΟΥ — runs/judge_l12.csv  [σύστημα MiniLM-L-12]")
        _judge_block(sub)
        print("    (επιλογή κατά ΡΙΣΚΟ: πόσες σελίδες άλλαξαν, όχι κατά ποσόστωση)")
    print("\n  -> Με το ταβάνι στο 48/50, ένα τρέξιμο κριτή ΔΕΝ μπορεί να δείξει")
    print("     βελτίωση, μόνο πτώση. Γι' αυτό ο L-12 ελέγχθηκε σε 12 και όχι σε 50.")


def sec_ragas():
    title("5.7  RAGAS  (πηγή: ragas_results.csv)")
    rows = read(os.path.join(HERE, "ragas_results.csv"))
    if not rows:
        missing("RAGAS", "ragas_env\\Scripts\\python.exe backend\\evaluation\\run_ragas.py")
        return
    print(f"  n={len(rows)}")
    for c in ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]:
        v = num(rows, c)
        if v:
            print(f"     {c:<20s} {mean(v):.4f}")


def sec_conv():
    title("5.6  ΣΥΝΟΜΙΛΙΕΣ  (πηγή: runs/conversational.csv)")
    rows = read(os.path.join(RUNS, "conversational.csv"))
    if not rows:
        missing("conversational", "evaluation/probe_conversational.py --csv ...")
        return
    real = [r for r in rows if (r.get("leak_test") or "").strip().lower()
            not in ("true", "1", "yes")]
    for key, label in [("cov_no_rw", "χωρίς rewrite"), ("cov_rw", "ΜΕ rewrite"),
                       ("cov_oracle", "oracle (ταβάνι)")]:
        print(f"  {label:<20s} {mean(num(real, key)):.1f}%")
    print(f"  leak tests: {len(rows) - len(real)}")


def sec_pages():
    title("5.6  ΑΛΛΑΞΕ ΤΟ PROMPT;  (πηγή: runs/pages_l12.csv)")
    rows = read(os.path.join(RUNS, "pages_l12.csv"))
    if not rows:
        missing("σύγκριση σελίδων", "evaluation/compare_pages_rerankers.py --csv ...")
        return
    same = sum(1 for r in rows
               if (r.get("same_set") or "").strip().lower() in ("true", "1", "yes"))
    print(f"  ίδιο σύνολο σελίδων: {same}/{len(rows)}"
          f"   ({len(rows) - same} ερωτήσεις πήραν ΑΛΛΕΣ σελίδες)")
    j = num(rows, "jaccard")
    if j:
        print(f"  μέσος Jaccard: {mean(j):.3f}")


def sec_manual():
    title("ΧΩΡΙΣ ARTIFACT — τρέξε την εντολή, μην αντιγράψεις παλιό νούμερο")
    missing("latency ανά στάδιο", "evaluation/measure_latency.py")
    missing("concurrency", "-e TORCH_THREADS=8 ... evaluation/concurrency_benchmark.py")
    missing("κόστος ανά 1.000", "evaluation/measure_cost.py --ratio 4.53")
    missing("scaling 418 -> 200k", "evaluation/scaling_benchmark.py")
    missing("πάτωμα τύχης", "evaluation/random_coverage_baseline.py")
    missing("διαστήματα εμπιστοσύνης", "evaluation/bootstrap_ci.py --compare A.csv B.csv")
    missing("corrective on/off", "evaluation/verify_corrective.py  (ΜΗ ΕΠΑΝΑΛΗΨΙΜΟ)")


SECTIONS = {
    "5.2": [sec_sets],
    "5.3": [sec_chunk, sec_candidates],
    "5.4": [sec_retrieval, sec_ladder, sec_gate],
    "5.5": [sec_judge],
    "5.6": [sec_conv, sec_pages],
    "5.7": [sec_ragas],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Νούμερα της διπλωματικής από τα artifacts.")
    ap.add_argument("--section", default=None,
                    help="μόνο μία ενότητα: " + ", ".join(sorted(SECTIONS)))
    args = ap.parse_args()

    if args.section:
        for fn in SECTIONS.get(args.section, []):
            fn()
        if args.section not in SECTIONS:
            print(f"Άγνωστη ενότητα. Διαθέσιμες: {', '.join(sorted(SECTIONS))}")
    else:
        for key in sorted(SECTIONS):
            for fn in SECTIONS[key]:
                fn()
        sec_manual()
    print()
