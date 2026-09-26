"""ΣΤΑΤΙΣΤΙΚΟΙ όροι ανά αρχείο — η ΔΩΡΕΑΝ εναλλακτική του probe_doc_terms.py.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ: στο probe_domain_glossary.py (νέα πεδία, καρδιολογία+ηφαιστειολογία)
η στατιστική μέθοδος `CD_constrained` έδωσε **ΤΟ ΙΔΙΟ** αποτέλεσμα με την LLM
`CG_specific`: coverage 95.0%, out_of_corpus 5/5 σιωπηλά. Ισοπαλία.

Στο ΚΥΡΙΟ σετ όμως η στατιστική μέθοδος ΔΕΝ έχει μετρηθεί ποτέ. Αν πιάσει κι αυτή
το 98.52% της CG, τότε το ingest ΔΕΝ χρειάζεται καθόλου κλήση Gemini:
  · μηδέν κόστος ανά upload
  · ντετερμινιστικό (ίδιο PDF -> ίδιοι όροι, πάντα)
  · μηδέν αποτυχία δικτύου σε ώρα upload
Αν βγει χαμηλότερα, η κλήση ανά αρχείο έχει μετρημένο λόγο ύπαρξης.

Γράφει το ΙΔΙΟ σχήμα JSON {filename: "όρος, όρος, ..."} με το probe_doc_terms.py,
ώστε να το καταναλώσει αυτούσιο το probe_glossary_main.py --terms-file.

ΚΟΣΤΟΣ: ΜΗΔΕΝ κλήσεις API. ~5 δευτερόλεπτα.

    docker compose exec backend python evaluation/dump_stat_terms.py \
        --out evaluation/runs/doc_terms_stat.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_domain_glossary import cleaned_terms

import ai_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/runs/doc_terms_stat.json")
    ap.add_argument("--n", type=int, default=15, help="όροι ανά αρχείο")
    args = ap.parse_args()

    got = ai_core.collection.get(include=["documents", "metadatas"])
    by_file: dict = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        by_file.setdefault(meta.get("file_name", "?"), []).append(doc)

    out = {}
    for fn in sorted(by_file):
        terms = cleaned_terms(by_file[fn], top=args.n)
        out[fn] = ", ".join(terms)
        print(f"{fn}  ({len(by_file[fn])} chunks)\n  {out[fn]}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
