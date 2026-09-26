"""Πόσο ΣΤΑΘΕΡΑ κόβονται οι ερωτήσεις εκτός σώματος; — συνέχεια της Φάσης 0.1.

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Στο eval_near_ooc η ερώτηση για τον GDPR (q048), που μέχρι τώρα κοβόταν «πάντα»,
    ΠΕΡΑΣΕ μέσω της δεύτερης ευκαιρίας (corrective). Όταν ο φύλακας κόβει, το
    Gemini ξαναγράφει την ερώτηση — και σε δύο τρεξίματα με 5 λεπτά απόσταση την
    ξανάγραψε διαφορετικά:
        'GDPR compliance obligations cloud service providers'                    -> -5.71 κομμένο
        'GDPR compliance obligations cloud providers data protection regulations' -> -3.40 ΠΕΡΑΣΕ
    Εδώ μετράμε (α) ΠΟΣΟ ΣΥΧΝΑ περνάει κάθε ερώτηση εκτός σώματος αν επαναλάβουμε το
    ίδιο πράγμα N φορές, και (β) τι ΑΠΑΝΤΑΕΙ το σύστημα όταν περνάει — γιατί το
    «πέρασε» δεν είναι από μόνο του διαρροή· διαρροή είναι μόνο αν η απάντηση
    ισχυριστεί κάτι για τον GDPR.

ΤΙ ΚΑΝΕΙ:
    • ίδιο απομονωμένο store με το eval_near_ooc (eval_near_ooc.open_store).
    • για κάθε ερώτηση: N φορές ο ΠΡΑΓΜΑΤΙΚΟΣ search_documents. Η μετάφραση είναι
      σταθερή (cache)· το 1ο πέρασμα είναι ντετερμινιστικό· η ΜΟΝΗ τυχαιότητα είναι η
      αναδιατύπωση του Gemini (temperature 0.1). Άρα το k/N μετράει ΑΚΡΙΒΩΣ αυτήν.
    • για κάθε ΔΙΑΦΟΡΕΤΙΚΗ αναδιατύπωση που πέρασε: μία απάντηση από την ask_ai +
      κριτής (ίδιος ορισμός με το eval_near_ooc) + μοτίβα άρνησης. Μία ανά διαφορετική,
      όχι ανά επανάληψη — ίδια αναδιατύπωση = ίδιες σελίδες.
    Ερωτήσεις: οι 5 out_of_corpus του golden_set_50 + οι 2 κοντινές που κόπηκαν
    (n033 OpenStack, n090 Open Container Initiative).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (25/9/2026):
    q048 περνάει 2-5/10 με ≥3 διαφορετικές αναδιατυπώσεις · q019 (best2 -4.47, 0.67
    κάτω από το όριο) 0-2/10 · q020/q049/q050/n033/n090 0/10.
    Όσες περάσουν: ο κριτής τις βγάζει ΟΛΕΣ «correct» (άρνηση) — όπως 38/40 κοντινές.

ΑΠΟΤΕΛΕΣΜΑ (25/9/2026, runs/corrective_flip.csv):
    πέρασαν 2/70 προσπάθειες: q048 1/10 (-2.88) · n090 1/10 (-2.11) · οι άλλες 0/10.
    Αναδιατύπωση ίδια ~9 στις 10 (temperature 0.1)· περνάει η σπάνια παραλλαγή.
    Απαντήσεις: 2/2 σωστή άρνηση. Πρόβλεψη: 2 σωστά στα 4. Ανάλυση στο AGENTS.md.

ΚΟΣΤΟΣ: 7 × N αναδιατυπώσεις (~100 tokens η καθεμία) + μία γέννηση (~10k) και μία
κλήση κριτή ανά διαφορετική αναδιατύπωση που περνάει. Χρόνος: ~1 λεπτό φόρτωση +
~3 s ανά επανάληψη ≈ 5 λεπτά με N=10.

    docker compose exec backend python evaluation/probe_corrective_flip.py
    docker compose exec backend python evaluation/probe_corrective_flip.py --n 20 --ids q048,q019
"""
import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_near_ooc as E

DEFAULT_IDS = "q019,q020,q048,q049,q050,n033,n090"
FIELDS = ["id", "attempt", "question", "translation", "rewrite", "best1", "best2", "outcome",
          "n_pages", "files", "answer", "refusal_regex", "judge", "judge_evidence",
          "evidence_found"]


def load_targets(ids: list[str]) -> list[dict]:
    """Γραμμές από τα δύο σετ, με focus για τον κριτή. Στο golden_set_50 δεν υπάρχει
    focus· το keyword του (quantum, gdpr, ...) είναι ακριβώς το θέμα που λείπει."""
    pool = {}
    for r in E.load_jsonl(E.GOLDEN_50):
        if r.get("category") == "out_of_corpus":
            pool[r["id"]] = {**r, "focus": r["keywords"][0], "question_en": r["question"]}
    for r in E.load_jsonl(E.NEAR_PATH):
        pool[r["id"]] = r
    missing = [i for i in ids if i not in pool]
    if missing:
        raise SystemExit(f"!! άγνωστα id: {missing}")
    return [pool[i] for i in ids]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="επαναλήψεις ανά ερώτηση")
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--no-answers", action="store_true", help="μόνο ανάκτηση, χωρίς γέννηση")
    ap.add_argument("--csv", default=os.path.join(E.HERE, "runs", "corrective_flip.csv"))
    args = ap.parse_args()
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY — σταματάω")
        return 1

    targets = load_targets([i.strip() for i in args.ids.split(",") if i.strip()])
    ai_core, _saved = E.open_store()
    print(f"gate {ai_core.MIN_RERANK_SCORE} · corrective {ai_core.CORRECTIVE_MIN_SCORE} · "
          f"corrective {'ΑΝΟΙΧΤΟΣ' if ai_core.ENABLE_CORRECTIVE else '!! ΚΛΕΙΣΤΟΣ'}")

    rows: list[dict] = []
    for t in targets:
        print(f"\n===== {t['id']} · {t['question'][:70]}")
        for k in range(1, args.n + 1):
            pages, b1, b2, rw, outcome = await E.retrieve(ai_core, t["question"])
            files = sorted({m.get("file_name", "?") for _x, m in pages})
            rows.append({
                "id": t["id"], "attempt": k, "question": t["question"],
                "translation": ai_core._translation_cache.get(t["question"], ""),
                "rewrite": rw, "best1": E.fmt(b1), "best2": E.fmt(b2), "outcome": outcome,
                "n_pages": len(pages), "files": ";".join(files), "_pages": pages, "_t": t})
            print(f"  {k:>2}  {outcome:<18} best2 {E.fmt(b2):>6}  {rw[:80]}", flush=True)

    # --- Επίπεδο 2: μία απάντηση ανά ΔΙΑΦΟΡΕΤΙΚΗ αναδιατύπωση που πέρασε ---
    if not args.no_answers:
        seen, todo = set(), []
        for r in rows:
            if r["outcome"] != "cut" and (r["id"], r["rewrite"]) not in seen:
                seen.add((r["id"], r["rewrite"]))
                todo.append(r)
        print(f"\n===== ΑΠΑΝΤΗΣΕΙΣ + ΚΡΙΤΗΣ σε {len(todo)} διαφορετικές αναδιατυπώσεις =====")
        for r in todo:
            t = r["_t"]
            r["answer"] = await E.answer_of(ai_core, t["question"], r["_pages"])
            r["refusal_regex"] = int(bool(E.REFUSAL.search(r["answer"])))
            try:
                label, evidence, found = await E.judge({**t, "answer": r["answer"]})
            except Exception as e:   # μία χαλασμένη ετυμηγορία δεν ρίχνει τις υπόλοιπες
                label, evidence, found = "?error", str(e)[:200], False
            r["judge"], r["judge_evidence"], r["evidence_found"] = label, evidence, int(found)
            print(f"  {r['id']} #{r['attempt']:<3} {label:<10} regex-άρνηση {r['refusal_regex']}"
                  f"  | {evidence[:110]}", flush=True)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- Σύνοψη ---
    print("\n" + "#" * 78)
    print(f"{'id':<6}{'πέρασε':>8}{'άνω όριο':>10}{'διαφ. αναδιατ.':>16}   best2 min..max   κριτής")
    total_pass = 0
    for t in targets:
        sub = [r for r in rows if r["id"] == t["id"]]
        k = sum(r["outcome"] != "cut" for r in sub)
        total_pass += k
        b2 = [float(r["best2"]) for r in sub if r["best2"]]
        rng = f"{min(b2):6.2f}..{max(b2):6.2f}" if b2 else "      —       "
        labels = [r.get("judge") for r in sub if r.get("judge")]
        verdict = ", ".join(f"{lab}×{labels.count(lab)}" for lab in sorted(set(labels))) or "—"
        print(f"{t['id']:<6}{k:>5}/{len(sub):<3}{E.upper_bound(k, len(sub)):>9.0%}"
              f"{len({r['rewrite'] for r in sub}):>12}        {rng}   {verdict}")
    n_all = len(rows)
    print(f"\nΣΥΝΟΛΟ: πέρασαν {total_pass}/{n_all} προσπάθειες "
          f"(άνω όριο 95% αν 0: {E.upper_bound(total_pass, n_all):.1%})")
    print("ΠΡΟΒΛΕΨΗ (γραμμένη πριν): q048 2-5/10 · q019 0-2/10 · οι άλλες 0/10 · "
          "όσες περάσουν -> όλες «correct».")
    print("ΕΠΟΜΕΝΟ: κάθε soft_leak/leak και κάθε διαφωνία κριτή/μοτίβων -> ΜΕ ΤΟ ΜΑΤΙ.")
    print(f"\nΓράφτηκε: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
