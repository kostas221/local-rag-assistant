"""Κόβει ο έλεγχος «όρος που ΔΕΝ υπάρχει στα έγγραφα» τις διαρροές του corrective
χωρίς να κόψει τις σωστές σώσεις; — ΜΗΔΕΝ κλήσεις Gemini.

ΑΦΟΡΜΗ (probe_corrective_domains.py, runs/corrective_domains.csv, 25/9/2026):
    Ο corrective agent ανεξάρτητος από πεδίο (prompts B/C) διέρρευσε 2/5 κοντινές
    ooc — και ο κανόνας Δ ΔΕΝ τις έπιασε (+1.4…+3.9, όσο οι σωστές σώσεις). Και
    στις ΤΕΣΣΕΡΙΣ διαρροές η αναδιατύπωση περιέχει λέξη με ΜΗΔΕΝ εμφανίσεις στα
    έγγραφα:
        o4  «Mount Vesuvius volcanic eruptions»          vesuvius = 0
        o5  «… HMG-CoA reductase inhibitors … lipid …»   hmg, coa, reductase, lipid = 0
        o5  «Which statins … cholesterol reduction?»     statins, cholesterol = 0
    Μια σωστή σώση κάνει το ΑΝΤΙΘΕΤΟ: αντικαθιστά αόριστες λέξεις με λέξεις που
    ΥΠΑΡΧΟΥΝ στο κείμενο («costing different amounts» -> «pricing»).

Η ΥΠΟΘΕΣΗ:
    Αν η αναδιατύπωση έχει ΕΣΤΩ ΕΝΑΝ όρο περιεχομένου με df = 0 στα in-scope
    έγγραφα, μένει κομμένη. Ντετερμινιστικό, μηδέν κόστος (το λεξιλόγιο το έχει
    ήδη το BM25), ανεξάρτητο από πεδίο — ρωτάει «υπάρχει αυτό που ψάχνεις;», όχι
    «μοιάζει με αυτό που ψάχνεις;» (που είναι ό,τι βαθμολογεί ο reranker).

ΟΡΙΣΜΟΣ ΓΡΑΜΜΕΝΟΣ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (δεν αλλάζει μετά):
    token      = ai_core.el_tokenize (ΙΔΙΟΣ με το BM25 index: lowercase, χωρίς
                 τόνους, \\w+, ΧΩΡΙΣ stemming).
    περιεχόμενο = token με >= 3 χαρακτήρες, όχι αριθμός, όχι στο
                 corpus_glossary._STOP (το stopword set της παραγωγής).
    κείμενο     = pymupdf + ai_core._normalize_pdf_text, ανά σελίδα — ΙΔΙΑ εξαγωγή
                 με το ingest.
    df = 0      = ο όρος δεν εμφανίζεται σε ΚΑΜΙΑ σελίδα των in-scope εγγράφων.

ΚΡΙΤΗΡΙΟ ΓΡΑΜΜΕΝΟ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ:
    ΠΕΡΝΑΕΙ μόνο αν (α) κόβει και τις 4 διαρροές του test-domains ΚΑΙ (β) ΔΕΝ κόβει
    καμία από τις σώσεις ΜΕ ΥΛΙΚΟ του cloud που περνούν σήμερα το −3.8 (h005, h008,
    h015). Μία κομμένη σωστή σώση = απόρριψη: θα έριχνε το hard set κάτω από 13/16.
    Το h016 (περνάει ΧΩΡΙΣ υλικό) αναφέρεται χωριστά — αν κοπεί, είναι μπόνους.

ΟΡΙΟ ΤΟΥ ΕΛΕΓΧΟΥ (γι' αυτό είναι ΑΝΑΓΚΑΙΟ, όχι ΙΚΑΝΟ):
    Οι αναδιατυπώσεις του cloud βγήκαν με το CLOUD prompt (A). Ο agent ανεξάρτητος
    από πεδίο θα χρησιμοποιούσε το B ή το C, που γράφουν ΑΛΛΕΣ αναδιατυπώσεις.
    Αν ο κανόνας περάσει εδώ, το επόμενο βήμα είναι το probe_corrective_domains.py
    στο cloud με B/C (κλήσεις Gemini). Αν αποτύχει εδώ, πέθανε με μηδέν κόστος.

    docker compose exec backend python evaluation/probe_oov_rewrite.py
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, "/app")

print("Φόρτωση ai_core (40-60 s χωρίς έξοδο) — ΜΗΝ το διακόψεις...", flush=True)

import pymupdf  # noqa: E402

import ai_core  # noqa: E402
import corpus_glossary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD_PAPERS = [
    "1702.04024.pdf", "1706.03178.pdf", "1812.03651.pdf", "1902.03383v1.pdf",
    "EECS-2009-28.pdf", "excamera-nsdi17.pdf", "mapreduce-osdi04.pdf",
]

# Ποιες σώσεις του cloud έχουν ΣΩΣΤΟ υλικό — από το AGENTS.md («μετρημένη κατανομή
# best2 (runs/corr_v1_control.csv): ΜΕ σωστό υλικό h015 · h005 · h008 · h009 ·
# ΧΩΡΙΣ υλικό h016 · h012»). Το CSV δεν έχει coverage για τα multi_hop.
CLOUD_MATERIAL = {"h005": True, "h008": True, "h009": True, "h015": True,
                  "h012": False, "h016": False}


def vocabulary(pdf_paths: list[str]) -> set[str]:
    vocab: set[str] = set()
    for p in pdf_paths:
        with pymupdf.open(p) as doc:
            for page in doc:
                vocab.update(ai_core.el_tokenize(ai_core._normalize_pdf_text(page.get_text() or "")))
    return vocab


def content_terms(text: str) -> list[str]:
    return [t for t in dict.fromkeys(ai_core.el_tokenize(text))
            if len(t) >= 3 and not t.isdigit() and t not in corpus_glossary._STOP]


def load_domains(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["skip"]:          # identical/error: ο agent δεν έτρεξε 2ο πέρασμα
                continue
            ooc = r["category"] == "out_of_corpus"
            rows.append({"set": "test_domains", "variant": r["variant"], "id": r["id"],
                         "ooc": ooc, "rewrite": r["rewrite"], "best2": float(r["best2"]),
                         "material": None if ooc else float(r["cov2"]) > 0})
    return rows


def load_cloud(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["best2"]:
                continue
            ooc = r["category"] == "out_of_corpus"
            rows.append({"set": "cloud", "variant": "A_cloud", "id": r["id"], "ooc": ooc,
                         "rewrite": r["q2"], "best2": float(r["best2"]),
                         "material": None if ooc else CLOUD_MATERIAL.get(r["id"])})
    return rows


def verdict(row: dict, passes: bool) -> str:
    if row["ooc"]:
        return "ΔΙΑΡΡΟΗ" if passes else "σιωπή"
    if not passes:
        return "κομμένο"
    return "ΣΩΘΗΚΕ" if row["material"] else "ΧΩΡΙΣ ΥΛΙΚΟ"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains-csv", default=os.path.join(HERE, "runs/corrective_domains.csv"))
    ap.add_argument("--cloud-csv", default=os.path.join(HERE, "runs/corr_v1_control.csv"))
    ap.add_argument("--floor", type=float, default=ai_core.CORRECTIVE_MIN_SCORE)
    args = ap.parse_args()

    dom_pdfs = sorted(glob.glob(os.path.join(HERE, "test_papers", "*.pdf")))
    cloud_pdfs = [os.path.join(HERE, "test_papers", "cloud", n) for n in CLOUD_PAPERS]
    missing = [p for p in cloud_pdfs if not os.path.exists(p)]
    if missing or len(dom_pdfs) != 2:
        print(f"!! Λείπουν PDF: {missing or dom_pdfs} — σταματάω")
        return 1
    vocabs = {"test_domains": vocabulary(dom_pdfs), "cloud": vocabulary(cloud_pdfs)}
    for k, v in vocabs.items():
        print(f"Λεξιλόγιο {k}: {len(v)} διακριτά tokens")

    rows = load_domains(args.domains_csv) + load_cloud(args.cloud_csv)
    print(f"\nκατώφλι 2ου περάσματος {args.floor} · κανόνας: 0 όροι με df=0\n")
    print(f"{'σετ':<13}{'prompt':<10}{'id':<5}{'best2':>7}  {'σήμερα':<12}{'+ OOV':<12}"
          "όροι με df=0 -> αναδιατύπωση")
    for r in rows:
        absent = [t for t in content_terms(r["rewrite"]) if t not in vocabs[r["set"]]]
        today = r["best2"] >= args.floor
        r["today"] = verdict(r, today)
        r["oov"] = verdict(r, today and not absent)
        r["absent"] = absent
        mark = "  <-- άλλαξε" if r["today"] != r["oov"] else ""
        print(f"{r['set']:<13}{r['variant']:<10}{r['id']:<5}{r['best2']:>+7.2f}  "
              f"{r['today']:<12}{r['oov']:<12}{','.join(absent) or '-'}{mark}\n"
              f"{'':<30}-> {r['rewrite'][:90]}")

    leaks_today = [r for r in rows if r["today"] == "ΔΙΑΡΡΟΗ"]
    leaks_oov = [r for r in rows if r["oov"] == "ΔΙΑΡΡΟΗ"]
    good_today = [r for r in rows if r["today"] == "ΣΩΘΗΚΕ"]
    good_lost = [r for r in good_today if r["oov"] != "ΣΩΘΗΚΕ"]
    nomat = [r for r in rows if r["today"] == "ΧΩΡΙΣ ΥΛΙΚΟ"]
    nomat_cut = [r for r in nomat if r["oov"] != "ΧΩΡΙΣ ΥΛΙΚΟ"]

    def ids(rs):
        return ", ".join(f"{r['id']}/{r['variant'][0]}" for r in rs) or "-"

    print("\n" + "#" * 78)
    print(f"διαρροές          σήμερα {len(leaks_today)} -> με OOV {len(leaks_oov)}   "
          f"(έμειναν: {ids(leaks_oov)})")
    print(f"σώσεις ΜΕ υλικό   σήμερα {len(good_today)} -> ΧΑΘΗΚΑΝ {len(good_lost)}   "
          f"({ids(good_lost)})")
    print(f"ΧΩΡΙΣ υλικό       σήμερα {len(nomat)} -> κόπηκαν {len(nomat_cut)}   "
          f"({ids(nomat_cut)})")
    ok = not leaks_oov and not good_lost
    print("\nΚΡΙΤΗΡΙΟ (γραμμένο πριν): 0 διαρροές ΚΑΙ 0 χαμένες σώσεις με υλικό -> "
          + ("ΠΕΡΝΑΕΙ — επόμενο: probe_corrective_domains στο cloud με B/C"
             if ok else "ΑΠΟΡΡΙΠΤΕΤΑΙ"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
