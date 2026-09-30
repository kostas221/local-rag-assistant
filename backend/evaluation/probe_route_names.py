"""ROUTING ΧΩΡΙΣ GEMINI: σπάσε την ερώτηση ΜΟΝΟ αν ονομάζει ≥ 2 έγγραφα — Φάση 2, βήμα 5, ΜΗΔΕΝ κόστος.

ΑΦΟΡΜΗ (probe_decomp_nonmh.py, 29/9/2026): το prompt διάσπασης του Αυγούστου σπάει ΣΧΕΔΟΝ ΤΑ ΠΑΝΤΑ
(κύριο 35/50, κοντινές ooc 36/42 — με παράφραση, όχι με θέματα). Η B δεν βλάπτει, αλλά θα πλήρωνε
+1 κλήση Gemini και ~+3 σελίδες σχεδόν σε ΚΑΘΕ ερώτηση για κέρδος που υπάρχει ΜΟΝΟ στις multi_hop.
Οι multi_hop των σετ ονομάζουν ΡΗΤΑ δύο papers -> ένας ντετερμινιστικός έλεγχος ονομάτων;

ΤΙ ΚΑΝΕΙ: διαβάζει ερώτηση + μετάφραση (runs/scoreboard/baseline.json, 8 σετ του cloud — τα «άλλα πεδία»
εκτός, άλλο σώμα) και μετράει πόσα ΔΙΑΦΟΡΕΤΙΚΑ έγγραφα ονομάζει, με τη λίστα ονομάτων του
scoreboard_answers.ALIASES. Σύγκριση με το LLM routing (decomp_mh40.csv, decomp_nonmh.csv) και με το
κέρδος της B (answers_decomp_mh_new.json). Τρέχει και χωρίς docker (μόνο αρχεία):
    python backend/evaluation/probe_route_names.py

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026 — ΧΩΡΙΣ γραμμένη πρόβλεψη: έλεγχος ad hoc, μετά το βήμα 4):
    σπάει:  mh_new 28/29 · mh_old 9/11 · κύριο 1/50 (μόνο multi_hop) · δύσκολο νέο 5/59 (μόνο multi_hop)
            · δύσκολο παλιό 0/16 · κοντινές ooc 0/42 · συνομιλίες 0/12 · πίνακες 0/16
    ΜΗΔΕΝ ψευδώς θετικά: καμία ερώτηση που δεν είναι multi_hop δεν σπάει (0/170).
    LLM (Αύγουστος) έναντι ονομάτων: κύριο 35 -> 1 · κοντινές 36 -> 0 · mh_new 29 -> 28 · mh_old 11 -> 9
    ΤΟ ΚΕΡΔΟΣ ΜΕΝΕΙ ΟΛΟ: mh_new σωστά μισά 43/58 με τον κανόνα, όσο και η B παντού (η m072 που χάνεται
        δεν είχε κέρδος). Σελίδες μ.ό.: κύριο 6.94 -> 7.02 (B παντού 9.40) · κοντινές 7.60 -> 7.60 (10.71).
    ΤΙ ΔΕΝ ΠΙΑΝΕΙ (multi_hop): «Berkeley view» σκέτο (διφορούμενο 2009/2019, δεν είναι στη λίστα),
        περιφράσεις («the video processing system», «the critical paper») — τα δύσκολα σετ είναι ΕΠΙΤΗΔΕΣ
        γραμμένα έτσι (hard_old 0/6, hard_new 5/14) — και αντωνυμίες (h011-h016).
    ⚠️ ΔΥΟ ΕΠΙΦΥΛΑΞΕΙΣ:
      (1) η λίστα ονομάτων γράφτηκε ΜΕ ΤΟ ΧΕΡΙ πάνω στις 29 του mh_new -> το 28/29 είναι αισιόδοξο· τα
          9/11 (mh_old) και 5/14 (hard_new) είναι πιο κοντά σε «εκτός δείγματος».
      (2) στην παραγωγή ΔΕΝ υπάρχει τέτοια λίστα για τα αρχεία του χρήστη: τα ονόματα (τίτλος, πρώτος
          συγγραφέας, όνομα συστήματος) πρέπει να βγαίνουν ΣΤΟ INGEST, ανά έγγραφο — ανοιχτό ζήτημα σχεδίασης.
    ΤΙ ΣΗΜΑΙΝΕΙ: με routing ονομάτων η ένσταση (β) του Αυγούστου («+1 κλήση σε ΚΑΘΕ ερώτηση») πέφτει —
        η κλήση γίνεται μόνο όταν ονομάζονται 2 έγγραφα. Και αφού ξέρουμε ΠΟΙΑ έγγραφα, ίσως δεν χρειάζεται
        καν η διάσπαση: αναζήτηση της ίδιας ερώτησης ΜΕΣΑ σε κάθε έγγραφο χωριστά (0 κλήσεις) — ΔΕΝ μετρήθηκε.
"""
import ast
import csv
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load_aliases() -> dict:
    """scoreboard_answers.ALIASES χωρίς import (εκείνο φορτώνει ai_core/μοντέλα)."""
    with open(os.path.join(HERE, "scoreboard_answers.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ALIASES")


def docs_named(text: str, aliases: dict) -> set:
    t = text.lower()
    return {d for d, al in aliases.items() if any(a in t for a in al)}


def main() -> int:
    aliases = load_aliases()
    with open(os.path.join(RUNS, "scoreboard", "baseline.json"), encoding="utf-8") as f:
        rows = [r for r in json.load(f)["rows"] if r["set"] != "domains"]
    named = {(r["set"], r["id"]): docs_named(f"{r['question']} || {r.get('translation') or ''}", aliases)
             for r in rows}
    routed = {k for k, v in named.items() if len(v) >= 2}

    print("ΚΑΝΟΝΑΣ ΟΝΟΜΑΤΩΝ: σπάει αν η ερώτηση (ή η μετάφρασή της) ονομάζει ≥ 2 έγγραφα")
    for sk in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == sk]
        cat = Counter(r["category"] for r in rs)
        cat2 = Counter(r["category"] for r in rs if (sk, r["id"]) in routed)
        print(f"  {sk:<9} σπάει {sum(cat2.values()):>2}/{len(rs):<3} "
              + " · ".join(f"{c} {cat2[c]}/{n}" for c, n in cat.items()))
    fp = [r for r in rows if r["category"] != "multi_hop" and (r["set"], r["id"]) in routed]
    non_mh = sum(r["category"] != "multi_hop" for r in rows)
    print(f"\nψευδώς θετικά (όχι multi_hop, σπάει): {len(fp)}/{non_mh}"
          + "".join(f"\n  {r['set']}:{r['id']} «{r['question'][:100]}»" for r in fp))
    print("\nmulti_hop που ΔΕΝ πιάνει:")
    for r in rows:
        if r["category"] == "multi_hop" and (r["set"], r["id"]) not in routed:
            print(f"  {r['set']}:{r['id']} ονόματα {sorted(named[(r['set'], r['id'])])} «{r['question'][:100]}»")

    llm = {}
    for fn in ("decomp_mh40.csv", "decomp_nonmh.csv"):
        with open(os.path.join(RUNS, fn), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                llm[(r["set"], r["id"])] = r
    print("\nLLM routing (Αύγουστος) έναντι ονομάτων:")
    for sk in ("mh_new", "mh_old", "main", "near_ooc"):
        ks = [k for k in llm if k[0] == sk]
        print(f"  {sk:<9} LLM σπάει {sum(int(llm[k]['n_sub']) > 1 for k in ks):>2}/{len(ks)} · "
              f"ονόματα {sum(k in routed for k in ks):>2}/{len(ks)}")

    with open(os.path.join(RUNS, "scoreboard", "answers_decomp_mh_new.json"), encoding="utf-8") as f:
        ans = json.load(f)["rows"]

    def ch(r):
        return (r["label_1"] == "correct") + (r["label_2"] == "correct")

    b0 = {r["id"]: ch(r) for r in ans if r["strategy"] == "base"}
    bb = {r["id"]: ch(r) for r in ans if r["strategy"] == "B"}
    kept = sum(bb[i] if ("mh_new", i) in routed else b0[i] for i in b0)
    print(f"\nmh_new σωστά μισά: βάση {sum(b0.values())} · B παντού {sum(bb.values())} · "
          f"B με κανόνα ονομάτων {kept} /{2 * len(b0)}")
    for sk in ("main", "near_ooc"):
        rs = [r for k, r in llm.items() if k[0] == sk]
        pg = [int(r["B_n"]) if (sk, r["id"]) in routed else int(r["base_n"]) for r in rs]
        print(f"{sk}: σελίδες μ.ό. βάση {sum(int(r['base_n']) for r in rs) / len(rs):.2f} · "
              f"B παντού {sum(int(r['B_n']) for r in rs) / len(rs):.2f} · B με κανόνα {sum(pg) / len(pg):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
