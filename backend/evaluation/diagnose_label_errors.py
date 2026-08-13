# backend/evaluation/diagnose_label_errors.py
"""ΠΟΙΟ ΑΠΟ ΤΑ ΔΥΟ ΠΕΔΙΑ ΛΑΘΕΨΕ ΤΟ ΜΟΝΤΕΛΟ ΣΤΗΝ ΕΤΙΚΕΤΑ;

Το probe_inline_citations.py --format label μέτρησε 10 ασύμφωνες παραπομπές
της μορφής «[S3, p.4]» ενώ το S3 είναι σελίδα 6 και η σελίδα 4 στάλθηκε ως S7.
Η μέτρηση λέει ΟΤΙ είναι ασύμφωνη, δεν λέει ΓΙΑΤΙ. Οι δύο εκδοχές οδηγούν σε
ΑΝΤΙΘΕΤΕΣ αποφάσεις:

  • σωστή ΣΕΛΙΔΑ / λάθος ΕΤΙΚΕΤΑ -> το μοντέλο αντιγράφει ό,τι βλέπει στο
    κείμενο αλλά ΔΕΝ παρακολουθεί τον τεχνητό δείκτη. Τότε η ετικέτα είναι
    λάθος ιδέα: κάθε «[Sn]» θα περνούσε τον έλεγχο ύπαρξης (υπάρχουν όλα τα
    S1..S8), δηλαδή η επαληθευσιμότητα ΚΑΤΑΡΓΕΙΤΑΙ σιωπηλά.
  • σωστή ΕΤΙΚΕΤΑ / λάθος ΣΕΛΙΔΑ -> η σελίδα είναι πλεονασμός. Τότε το σκέτο
    «[S3]» είναι ΚΑΙ σωστό ΚΑΙ ακόμα φθηνότερο.

ΠΩΣ ΚΡΙΝΕΤΑΙ, ΧΩΡΙΣ ΚΡΙΤΗ: για κάθε ασύμφωνη παραπομπή συγκρίνεται η
λεξιλογική επικάλυψη της πρότασης με (α) τη σελίδα ΤΗΣ ΕΤΙΚΕΤΑΣ που έγραψε
και (β) τη σελίδα ΤΟΥ ΑΡΙΘΜΟΥ που έγραψε. Όποια είναι υψηλότερη, εκείνο το
πεδίο δείχνει την πηγή που όντως χρησιμοποίησε. Το πάτωμα τύχης είναι ο μέσος
όρος των ΥΠΟΛΟΙΠΩΝ σελίδων — αν καμία από τις δύο δεν το ξεπερνά, η πρόταση
δεν στηρίζεται σε καμία από τις δύο και η παραπομπή είναι απλώς λάθος.

ΜΗΔΕΝ κλήσεις γέννησης: ξανατρέχει μόνο η ανάκτηση (ντετερμινιστική).
ΠΡΟΣΟΧΗ: για το h016 τρέχει ο corrective agent, που ΔΕΝ είναι επαναλήψιμος —
αν οι σελίδες του διαφέρουν από το τρέξιμο του probe, παραλείπεται.

Τρέξε:
  docker compose exec backend python evaluation/diagnose_label_errors.py \
      --csv evaluation/runs/inline_citations_label.csv
"""
import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_inline_citations import (
    PLACEHOLDER_RE,
    content_tokens,
    file_key,
    has_greek,
    load_subset,
    overlap,
    parse_citations,
    sentence_before,
    strip_citations,
)

import ai_core


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/runs/inline_citations_label.csv")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == "cite"]
    answers = {r["id"]: r["answer"] for r in rows}

    # Τα ίδια ids με το probe -> ίδιο ερώτημα, ίδια ανάκτηση.
    subset = {r["id"]: r for r in load_subset(20, ["h016"])}

    tally = {"page": 0, "label": 0, "καμία": 0, "ισοπαλία": 0}
    print(f"{'id':6s} {'γράφτηκε':12s} {'ετικέτα->σελ':13s} "
          f"{'επικ.ετικ':9s} {'επικ.σελ':9s} {'πάτωμα':7s} ετυμηγορία")
    print("-" * 84)

    for qid, answer in answers.items():
        if qid not in subset:
            continue
        q = subset[qid]["question"]
        is_gr = has_greek(q)
        pre = await ai_core.search_documents(q, None, user_id=None)
        # Ίδια αρίθμηση με το ai_core: enumerate(top_3_data, 1).
        pages = [(f"s{i}", file_key(m.get("file_name", "?")),
                  str(m.get("page", "?")), t)
                 for i, (t, m) in enumerate(pre, 1)]
        by_label = {p[0]: p for p in pages}
        by_page = {}
        for p in pages:
            by_page.setdefault(p[2], []).append(p)

        cites, _ = parse_citations(answer)
        for start, raw, page in cites:
            if PLACEHOLDER_RE.search(raw):
                continue
            lab = file_key(raw)
            if lab in by_label and by_label[lab][2] == page:
                continue                      # σύμφωνη — δεν μας απασχολεί
            if lab not in by_label:
                continue                      # ανύπαρκτη ετικέτα, άλλη κλάση

            sent = strip_citations(sentence_before(answer, start))
            toks = content_tokens(sent, latin_only=is_gr)
            if len(toks) < 2:
                continue

            o_lab = overlap(toks, by_label[lab][3])
            cand = by_page.get(page, [])
            o_pag = max((overlap(toks, c[3]) for c in cand), default=float("nan"))
            others = [p for p in pages
                      if p[0] != lab and p not in cand]
            floor = (sum(overlap(toks, p[3]) for p in others) / len(others)
                     if others else float("nan"))

            if o_pag != o_pag or o_lab > o_pag:
                verdict = "ΕΤΙΚΕΤΑ σωστή, σελίδα λάθος"
                tally["label"] += 1
            elif o_pag > o_lab:
                verdict = "ΣΕΛΙΔΑ σωστή, ετικέτα λάθος"
                tally["page"] += 1
            else:
                verdict = "ισοπαλία — αναποφάσιστο"
                tally["ισοπαλία"] += 1
            if max(o_lab, o_pag if o_pag == o_pag else 0) <= floor:
                verdict = "ΚΑΜΙΑ από τις δύο (κάτω από το πάτωμα)"
                tally["καμία"] += 1

            print(f"{qid:6s} [{lab.upper()}, p.{page}]".ljust(19)
                  + f"{lab.upper()}->σελ {by_label[lab][2]:<4s} "
                  + f"{o_lab:9.2f} "
                  + (f"{o_pag:9.2f} " if o_pag == o_pag else f"{'—':>9s} ")
                  + f"{floor:7.2f} " + verdict)

    print("\n--- ΠΟΡΙΣΜΑ ---")
    print(f"  σωστή ΣΕΛΙΔΑ / λάθος ετικέτα : {tally['page']}")
    print(f"  σωστή ΕΤΙΚΕΤΑ / λάθος σελίδα : {tally['label']}")
    print(f"  ισοπαλία                     : {tally['ισοπαλία']}")
    print(f"  καμία (κάτω από το πάτωμα)   : {tally['καμία']}")
    print("\n  Αν κυριαρχεί η ΣΕΛΙΔΑ: το μοντέλο αντιγράφει, δεν δεικτοδοτεί ->")
    print("  η ετικέτα καταργεί την επαληθευσιμότητα (κάθε Sn υπάρχει).")
    print("  Αν κυριαρχεί η ΕΤΙΚΕΤΑ: η σελίδα είναι πλεονασμός -> σκέτο [S3].")


if __name__ == "__main__":
    asyncio.run(main())
