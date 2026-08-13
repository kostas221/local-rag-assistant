"""Έλεγχος ακεραιότητας του `golden_hard_paraphrase.jsonl` — ΜΗΔΕΝ κόστος.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ
-------------
Το hard set είναι **ο μόνος οδηγός βελτίωσης που έχει απομείνει** (τα άλλα
τρία είναι στο ταβάνι). Κάθε συμπέρασμα για τις τέσσερις άλυτες στηρίζεται σε
αυτό. Αν το ίδιο το σετ έχει λάθος, τα συμπεράσματα είναι λάθος — και το
`verify_keywords.py` ελέγχει ΜΟΝΟ ύπαρξη/σπανιότητα, όχι αν η παράφραση
αντιστοιχεί στη γονική.

ΤΙ ΕΛΕΓΧΕΙ (τρία ξεχωριστά πράγματα)
------------------------------------
1. **Γονική ερώτηση**: υπάρχει; και σε ΠΟΙΟ σετ; Μια παράφραση με ανύπαρκτη
   γονική έχει keywords που δεν κληρονομήθηκαν από επαληθευμένη ερώτηση.
2. **Keywords έναντι γονικής**: ταυτόσημα, υποσύνολο ή ΞΕΝΑ; Ένα ξένο keyword
   σημαίνει ότι κάποιος τα έγραψε από την αρχή — δεν ισχύει το «ήδη
   επαληθευμένα» που δικαιολογεί όλο το σετ.
3. **Ονομάζει η παράφραση τον στόχο;** Αν η γονική λέει «ExCamera» και η
   παράφραση λέει «that system paper», τα keywords απαιτούν ΕΝΑ paper ενώ η
   ερώτηση δεν το προσδιορίζει. Τότε η αποτυχία δεν είναι αδυναμία του
   συστήματος, είναι **ανεκπλήρωτο αναφορικό** — και το coverage μετράει άλλο
   πράγμα από αυτό που ρωτάει η ερώτηση.

    docker compose exec backend python evaluation/audit_hard_set.py
"""
import argparse
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

EVAL = "/app/evaluation"
HARD = "golden_hard_paraphrase.jsonl"

# Όλα τα σετ όπου μπορεί να ζει μια γονική ερώτηση.
PARENT_SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
               "golden_tables.jsonl", "golden_conversations.jsonl"]

# Ονόματα συστημάτων/paper που εμφανίζονται στο corpus. Αν η γονική τα λέει
# και η παράφραση όχι, ο στόχος δεν προσδιορίζεται από την ερώτηση.
ENTITIES = ["excamera", "pywren", "mapreduce", "sprocket", "gg", "lambda",
            "aws", "gvisor", "firecracker", "vpxenc", "berkeley", "quincy",
            "idaho", "hadoop", "dynamo", "s3", "ec2", "faas", "baas"]

DEIC = re.compile(
    r"\b(that|those|the other|the same|this particular|such|it|they)\b"
    r"|\b(εκειν|συγκεκριμεν|αυτ|αλλ)\w*", re.I)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def load(fname):
    path = os.path.join(EVAL, fname)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("id")
        if not qid:                      # γραμμές χωρίς id: τις δείχνουμε
            print(f"  ! {fname}: γραμμή χωρίς id -> {str(d)[:70]}")
            continue
        d["_set"] = fname.split(".")[0]
        out[qid] = d
    return out


def entities_in(text):
    t = strip_accents(text or "").lower()
    return {e for e in ENTITIES if re.search(rf"\b{re.escape(e)}\b", t)}


def main(args) -> int:
    parents = {}
    for f in PARENT_SETS:
        for k, v in load(f).items():
            parents.setdefault(k, v)
    hard = load(HARD)
    print(f"\n{len(hard)} παραφράσεις · {len(parents)} υποψήφιες γονικές "
          f"από {len(PARENT_SETS)} σετ\n")

    problems = {"ορφανή": [], "ξένα-kw": [], "χαμένη-οντότητα": [],
                "δεικτικό": []}

    for hid, h in sorted(hard.items()):
        pid = h.get("parent", "")
        p = parents.get(pid)
        hkw = [k.lower() for k in h.get("keywords", [])]
        hq = h.get("question", "")
        print("=" * 78)
        print(f"{hid}  [{h.get('hard_type','?'):<9}] γονική {pid or '—'}"
              + (f"  ({p['_set']})" if p else "  ΔΕΝ ΒΡΕΘΗΚΕ ΠΟΥΘΕΝΑ"))
        print(f"  παράφραση : {hq}")
        print(f"  keywords  : {hkw}")

        if not p:
            problems["ορφανή"].append(hid)
            print("  ** ΟΡΦΑΝΗ: τα keywords δεν κληρονομήθηκαν από "
                  "επαληθευμένη ερώτηση")
        else:
            pkw = [k.lower() for k in p.get("keywords", [])]
            pq = p.get("question", "")
            print(f"  γονική    : {pq}")
            print(f"  keywords γ: {pkw}")
            extra = [k for k in hkw if k not in pkw]
            missing = [k for k in pkw if k not in hkw]
            if extra:
                problems["ξένα-kw"].append(hid)
                rel = "ΞΕΝΑ +" + ",".join(extra)
            elif missing:
                rel = "υποσύνολο (−" + ",".join(missing) + ")"
            else:
                rel = "ΤΑΥΤΟΣΗΜΑ"
            print(f"  σχέση kw  : {rel}")

            lost = entities_in(pq) - entities_in(hq)
            if lost:
                problems["χαμένη-οντότητα"].append(hid)
                print(f"  ** Η ΓΟΝΙΚΗ ΟΝΟΜΑΖΕΙ, Η ΠΑΡΑΦΡΑΣΗ ΟΧΙ: "
                      f"{sorted(lost)}")

        if DEIC.search(strip_accents(hq)):
            problems["δεικτικό"].append(hid)
            print("  ** δεικτικό/αντωνυμία στην παράφραση")

    print("=" * 78)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ")
    for k, v in problems.items():
        print(f"  {k:<18}{len(v):>3}  {sorted(set(v))}")
    print("-" * 78)
    print("ΤΟ ΚΡΙΣΙΜΟ: «χαμένη οντότητα» + «δεικτικό» μαζί σημαίνει ότι η")
    print("ερώτηση ΔΕΝ προσδιορίζει τον στόχο που απαιτούν τα keywords της.")
    print("Η σιωπή εκεί είναι ΣΩΣΤΗ απάντηση, όχι αποτυχία ανάκτησης.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ακεραιότητα του hard set")
    raise SystemExit(main(ap.parse_args()))
