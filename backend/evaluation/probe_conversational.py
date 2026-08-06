"""Δουλεύει το conversational query rewriting; — ΠΟΤΕ ΔΕΝ ΜΕΤΡΗΘΗΚΕ.

ΤΟ ΚΕΝΟ:
Το `_rewrite_query` τρέχει σε ΚΑΘΕ follow-up ερώτηση στην παραγωγή («και το
κόστος του;» -> αυτόνομο ερώτημα) και ΚΑΝΕΝΑ από τα golden sets δεν έχει
πολύγυρες συνομιλίες. Δηλαδή: agentic λειτουργία που ήδη σερβίρει χρήστες,
χωρίς μία γραμμή μέτρησης. Ίδιο μοτίβο με το gate που έδειχνε «61/61 τέλειο».

ΤΡΕΙΣ ΜΕΤΡΗΣΕΙΣ ΑΝΑ ΣΥΝΟΜΙΛΙΑ (η σύγκριση είναι το παν):
  NO-REWRITE  το follow-up σκέτο, χωρίς ιστορικό -> τι θα γινόταν ΧΩΡΙΣ τη
              λειτουργία. Το κάτω όριο.
  REWRITE     το follow-up + ιστορικό μέσα από το ΠΡΑΓΜΑΤΙΚΟ _rewrite_query.
  ORACLE      η πλήρης, αυτόνομη ερώτηση-γονέας -> το ταβάνι. Δείχνει πόσο από
              το κενό καλύπτει το rewrite· χωρίς αυτό, ένα «60%» δεν λέει αν
              είναι καλό ή κακό.

ΔΥΟ LEAK TESTS (`leak_test: true`): follow-up εκτός θέματος μετά από συνομιλία
εντός θέματος. Το rewrite ΔΕΝ πρέπει να κολλήσει το cloud context πάνω στο
«ποια είναι η τιμή του Bitcoin;» και να το περάσει από το gate. Αν διαρρεύσει,
η λειτουργία υπονομεύει την άμυνα κατά της ψευδαίσθησης σε κάθε συνομιλία.

Τα keywords έρχονται από ΕΠΑΛΗΘΕΥΜΕΝΗ ερώτηση-γονέα (ίδιο μοτίβο με το hard
set) -> αν το coverage πέσει, ξέρουμε ότι φταίει η ΔΙΑΤΥΠΩΣΗ, όχι τα keywords.

ΚΟΣΤΟΣ: 1 κλήση rewrite ανά συνομιλία (12) + μεταφράσεις για νέες ελληνικές.
Οι απαντήσεις του 1ου γύρου είναι ΓΡΑΜΜΕΝΕΣ στο σετ — καμία κλήση γέννησης,
και το ιστορικό μένει ντετερμινιστικό μεταξύ εκτελέσεων.

    docker compose exec backend python evaluation/probe_conversational.py
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

SET = "/app/evaluation/golden_conversations.jsonl"
PARENTS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
]


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_parents():
    out = {}
    for p in PARENTS:
        if os.path.exists(p):
            for t in load(p):
                out.setdefault(t["id"], t)
    return out


def covered(pages, kws):
    blob = "\n".join(t for t, _m in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    convs = load(SET)
    parents = load_parents()
    print(f"{len(convs)} συνομιλίες · {len(convs)} κλήσεις rewrite\n")

    print(f"{'id':<6} {'parent':<6} {'category':<14} {'no-rw':>6} {'rw':>6} "
          f"{'oracle':>7}  {'έκβαση'}")
    print("-" * 74)

    rows = []
    for c in convs:
        kws = c["keywords"]
        n = len(kws)
        leak = c.get("leak_test", False)

        # A. χωρίς ιστορικό — ό,τι θα έβλεπε το σύστημα αν δεν υπήρχε rewrite
        no_rw = await ai_core.search_documents(c["followup"], GOLDEN_CORPUS)

        # B. με ιστορικό, μέσα από τον ΠΡΑΓΜΑΤΙΚΟ κώδικα παραγωγής
        history = [
            {"role": "user", "content": c["turn1_question"]},
            {"role": "assistant", "content": c["turn1_answer"]},
        ]
        rewritten = await ai_core._rewrite_query(c["followup"], history)
        with_rw = await ai_core.search_documents(rewritten, GOLDEN_CORPUS)

        # C. ταβάνι — η πλήρης αυτόνομη ερώτηση
        parent = parents.get(c["parent"])
        oracle = (await ai_core.search_documents(parent["question"], GOLDEN_CORPUS)
                  if parent else [])

        cov_a, cov_b, cov_c = (covered(no_rw, kws), covered(with_rw, kws),
                               covered(oracle, kws))

        if leak:
            outcome = "ΔΙΑΡΡΟΗ" if with_rw else "σιωπή (σωστό)"
        elif cov_b > cov_a:
            outcome = "ΒΟΗΘΗΣΕ"
        elif cov_b < cov_a:
            outcome = "ΕΒΛΑΨΕ"
        else:
            outcome = "ουδέτερο"

        rows.append(dict(id=c["id"], parent=c["parent"],
                         category=c.get("category", ""), leak_test=leak, n_kw=n,
                         cov_no_rw=cov_a, cov_rw=cov_b, cov_oracle=cov_c,
                         pages_no_rw=len(no_rw), pages_rw=len(with_rw),
                         pages_oracle=len(oracle),
                         followup=c["followup"], rewritten=rewritten))
        print(f"{c['id']:<6} {c['parent']:<6} {c.get('category',''):<14} "
              f"{f'{cov_a}/{n}':>6} {f'{cov_b}/{n}':>6} {f'{cov_c}/{n}':>7}  {outcome}",
              flush=True)

    inc = [r for r in rows if not r["leak_test"]]
    leaks = [r for r in rows if r["leak_test"]]

    print("\n" + "=" * 74)
    print("COVERAGE (in-corpus συνομιλίες)")
    print("=" * 74)
    tot = sum(r["n_kw"] for r in inc)
    a = 100 * sum(r["cov_no_rw"] for r in inc) / tot
    b = 100 * sum(r["cov_rw"] for r in inc) / tot
    c_ = 100 * sum(r["cov_oracle"] for r in inc) / tot
    print(f"  χωρίς rewrite (κάτω όριο)   {a:>6.1f}%")
    print(f"  ΜΕ rewrite   (παραγωγή)     {b:>6.1f}%   ({b-a:+.1f} έναντι του κάτω ορίου)")
    print(f"  oracle       (ταβάνι)       {c_:>6.1f}%")
    if c_ > a:
        closed = 100 * (b - a) / (c_ - a)
        print(f"\n  Το rewrite καλύπτει το {closed:.0f}% του κενού μεταξύ "
              f"κάτω ορίου και ταβανιού.")
    elif abs(c_ - a) < 0.05:
        print("\n  ΠΡΟΣΟΧΗ: κάτω όριο == ταβάνι -> το σετ ΔΕΝ δοκιμάζει τίποτα. "
              "Τα follow-up είναι πολύ αυτόνομα· ξαναγράψ' τα πιο ελλειπτικά.")

    print("\n" + "=" * 74)
    print("LEAK TESTS — μολύνει το ιστορικό τα εκτός θέματος ερωτήματα;")
    print("=" * 74)
    for r in leaks:
        status = "ΔΙΑΡΡΟΗ" if r["pages_rw"] else "σιωπή (σωστό)"
        print(f"  {r['id']}: {status}   (χωρίς rewrite: "
              f"{'απάντησε' if r['pages_no_rw'] else 'σιωπή'})")
        print(f"     follow-up:  {r['followup']}")
        print(f"     rewritten:  {r['rewritten']}")

    helped = [r for r in inc if r["cov_rw"] > r["cov_no_rw"]]
    hurt = [r for r in inc if r["cov_rw"] < r["cov_no_rw"]]
    silent = [r for r in inc if r["pages_rw"] == 0]
    print(f"\n  ΒΟΗΘΗΣΕ: {len(helped)}" + (f" ({', '.join(r['id'] for r in helped)})" if helped else ""))
    print(f"  ΕΒΛΑΨΕ:  {len(hurt)}" + (f" ({', '.join(r['id'] for r in hurt)})" if hurt else ""))
    print(f"  σιωπηλές μετά το rewrite: {len(silent)}"
          + (f" ({', '.join(r['id'] for r in silent)})" if silent else " — το gate δεν χτύπησε ποτέ"))

    print("\n  ΤΙ ΠΑΡΗΓΑΓΕ ΤΟ REWRITE:")
    for r in inc:
        flag = ""
        if r["cov_rw"] > r["cov_no_rw"]:
            flag = "  [+]"
        elif r["cov_rw"] < r["cov_no_rw"]:
            flag = "  [-]"
        print(f"    {r['id']}{flag}  '{r['followup'][:52]}'")
        print(f"           -> '{r['rewritten']}'")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
