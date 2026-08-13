"""Ανίχνευση ΑΣΑΦΟΥΣ ΕΡΩΤΗΣΗΣ πριν το retrieval — regex ΕΝΑΝΤΙΟΝ agent.

ΤΟ ΕΡΩΤΗΜΑ
----------
Οι τέσσερις άλυτες του hard set δεν έχουν κοινό αίτιο ανάκτησης. Δύο από αυτές
(`h012`, `h016`) είναι ερωτήσεις με **δεικτικό χωρίς προηγούμενο** («η
συγκεκριμένη υλοποίηση», «that older system»). Καμία μηχανική κατάντη της
ερώτησης δεν τις λύνει γιατί δεν λείπει υλικό, λείπει το ΥΠΟΚΕΙΜΕΝΟ.

Όλοι οι μηχανισμοί που δοκιμάστηκαν ως τώρα κοιτούσαν **σελίδες**: κριτής
ΝΑΙ/ΟΧΙ, quote-first, RBO, NLI. Η ΕΡΩΤΗΣΗ είναι η μόνη είσοδος που δεν έχει
κοιτάξει κανείς.

ΤΙ ΣΥΓΚΡΙΝΕΤΑΙ
--------------
  regex : ντετερμινιστικό λεξικό δεικτικών. ΜΗΔΕΝ κόστος, ΜΗΔΕΝ latency,
          απολύτως επαναλήψιμο. Πιάνει ΜΟΡΦΗ.
  agent : μία κλήση Gemini ανά ερώτηση. Πιάνει ΝΟΗΜΑ, αλλά κοστίζει +1 κλήση
          σε ΚΑΘΕ ερώτηση της παραγωγής — το ίδιο κόστος που απέρριψε ήδη
          decomposition, enrichment, grounding verdict και ευαίσθητο trigger.

ΤΟ ΚΡΙΤΗΡΙΟ ΔΕΝ ΕΙΝΑΙ «ΠΟΣΕΣ ΠΙΑΝΕΙ»
------------------------------------
Είναι τα **ψευδώς θετικά**. Ένας ανιχνευτής που ρωτάει διευκρίνιση σε ερώτηση
που σήμερα παίρνει 5/5/5/5 χαλάει κάτι που δουλεύει. Το `grounding_verdict`
πέθανε ακριβώς έτσι: 4 λανθασμένες αρνήσεις για 1 κερδισμένη.

Ο ΚΑΝΟΝΑΣ ΤΟΥ ΙΣΤΟΡΙΚΟΥ
-----------------------
Στην παραγωγή το `_rewrite_query` επιλύει αναφορικά από το ιστορικό με
μετρημένη κάλυψη 90% έναντι 41.2% τύχης. Άρα ο ανιχνευτής έχει νόημα ΜΟΝΟ
όταν ΔΕΝ υπάρχει ιστορικό. Το `golden_conversations` περνάει με ιστορικό και
οφείλει να δώσει ΜΗΔΕΝ σημάνσεις — αν δώσει, ο ανιχνευτής θα διέκοπτε
συνομιλίες που σήμερα δουλεύουν.

    docker compose exec backend python evaluation/probe_ambiguity_detector.py \\
        --csv evaluation/runs/ambiguity.csv              # regex μόνο, ΔΩΡΕΑΝ
    docker compose exec backend python evaluation/probe_ambiguity_detector.py \\
        --agent --csv evaluation/runs/ambiguity.csv      # + 89 κλήσεις, ~0.05 $
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

EVAL = "/app/evaluation"

# Τα σετ και το αν η ερώτηση φτάνει στο σύστημα ΜΕ ιστορικό. Οι συνομιλίες
# έχουν, άρα εκεί ο ανιχνευτής ΔΕΝ πρέπει να τρέχει καν στην παραγωγή — τις
# κρατάμε ως έλεγχο ψευδώς θετικών.
SETS = [
    ("golden_set_50.jsonl", False),
    ("golden_multihop_new.jsonl", False),
    ("golden_hard_paraphrase.jsonl", False),
    ("golden_tables.jsonl", False),
    ("golden_conversations.jsonl", True),
]

# Οι τέσσερις που το σύστημα αφήνει άλυτες σήμερα.
UNSOLVED = {"h008", "h009", "h012", "h016"}

# Δεικτικό + ουσιαστικό οντότητας. Το ουσιαστικό είναι απαραίτητο: σκέτο
# «that» είναι και σύνδεσμος («the paper that describes…») και θα έδινε
# ψευδώς θετικά σε κάθε δεύτερη ερώτηση.
DEIC_EN = re.compile(
    r"\b(that|those|the other|the same|this particular|such)\s+"
    r"(\w+\s+){0,2}"
    r"(paper|system|implementation|work|approach|study|model|one|thing|"
    r"technique|method|tool|service|experiment|result|author)s?\b", re.I)

DEIC_EL = re.compile(
    r"\b(εκειν|συγκεκριμεν|αυτ|αλλ)\w*\s+(τ\w+\s+)?"
    r"(υλοποιηση|paper|συστημα|εργασια|μελετη|μεθοδο|προσεγγιση|"
    r"πειραμα|αποτελεσμα|εργαλειο)", re.I)

PROMPT = """You are a query triage step in a document question-answering system.
The user asks a question with NO conversation history available.

Decide if the question can be retrieved against a document collection AS IT
STANDS, or if it contains a reference to something that was never introduced
(a demonstrative like "that system", "the other paper", "this implementation",
or a pronoun whose antecedent is missing).

Answer with exactly one word on the first line: CLEAR or AMBIGUOUS
If AMBIGUOUS, add a second line with the single clarifying question you would
ask the user, under 15 words.

Be conservative. A question that is merely broad, vague or non-technical is
CLEAR. Only mark AMBIGUOUS when a specific unnamed entity is being referred to.

QUESTION: {q}"""


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def question_of(d: dict) -> str:
    """Τα σετ δεν έχουν ίδιο σχήμα — το conversations έχει followup."""
    return (d.get("question") or d.get("followup")
            or d.get("turn1_question") or "")


def detect_regex(q: str) -> bool:
    return bool(DEIC_EN.search(q) or DEIC_EL.search(strip_accents(q)))


async def detect_agent(q: str):
    """Επιστρέφει (ασαφής;, διευκρινιστική).

    ΤΟ ΣΦΑΛΜΑ ΔΕΝ ΚΑΤΑΠΙΝΕΤΑΙ. Η πρώτη εκδοχή είχε `except` που γύριζε None,
    και επειδή το `generate_once` θέλει `model`/`api_key` ως keyword-only, ΚΑΘΕ
    κλήση πετούσε TypeError: το probe τύπωσε καθαρό 0/105 σαν να ήταν εύρημα.
    Ένας σιωπηλός ανιχνευτής και ένας ανιχνευτής που δεν βρίσκει τίποτα δίνουν
    ΤΟ ΙΔΙΟ νούμερο — γι' αυτό εδώ σκάει.
    """
    import ai_core
    from gemini_rest import generate_once
    out = (await generate_once(PROMPT.format(q=q),
                               model=ai_core.GEMINI_MODEL,
                               api_key=ai_core.GEMINI_API_KEY,
                               thinking_budget=0,
                               max_output_tokens=64)).strip()
    first, _, rest = out.partition("\n")
    return first.strip().upper().lstrip("*# ").startswith("AMBIG"), \
        rest.strip()[:120]


def load_all():
    rows = []
    for fname, has_history in SETS:
        path = os.path.join(EVAL, fname)
        if not os.path.exists(path):
            print(f"λείπει {fname} — παραλείπεται")
            continue
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        for line in lines:
            if not line.strip():
                continue
            d = json.loads(line)
            q = question_of(d)
            if not q:
                continue
            rows.append({"id": d.get("id", "?"), "set": fname.split(".")[0],
                         "has_history": has_history, "question": q})
    return rows


async def main(args) -> int:
    rows = load_all()
    print(f"{len(rows)} ερωτήσεις από {len({r['set'] for r in rows})} σύνολα")
    print(f"ανιχνευτές: regex{' + agent' if args.agent else ''}\n")

    for i, r in enumerate(rows, 1):
        r["regex"] = detect_regex(r["question"])
        if args.agent:
            flag, ask = await detect_agent(r["question"])
            r["agent"] = flag
            r["clarify"] = ask
            print(f"  [{i}/{len(rows)}] {r['id']}", end="\r", flush=True)
            if args.delay:
                await asyncio.sleep(args.delay)
        else:
            r["agent"] = None
            r["clarify"] = ""
    if args.agent:
        print(" " * 40)

    dets = ["regex"] + (["agent"] if args.agent else [])
    print("=" * 78)
    print(f"{'σύνολο':<26}{'n':>5}" + "".join(f"{d:>10}" for d in dets))
    print("-" * 78)
    for fname, _h in SETS:
        s = fname.split(".")[0]
        grp = [r for r in rows if r["set"] == s]
        if not grp:
            continue
        cells = "".join(f"{sum(1 for r in grp if r[d]):>10}" for d in dets)
        print(f"{s:<26}{len(grp):>5}{cells}")
    print("-" * 78)

    # Ψευδώς θετικά: οτιδήποτε σημαδεύεται ΕΚΤΟΣ των τεσσάρων άλυτων.
    for d in dets:
        fp = [r for r in rows if r[d] and r["id"] not in UNSOLVED]
        tp = sorted(r["id"] for r in rows if r[d] and r["id"] in UNSOLVED)
        conv = [r for r in fp if r["has_history"]]
        print(f"\n{d.upper()}")
        print(f"  άλυτες που πιάστηκαν   {len(tp)}/4  {tp}")
        print(f"  ψευδώς θετικά          {len(fp)}"
              + ("  <== ΚΟΣΤΟΣ ΣΕ ΕΡΩΤΗΣΕΙΣ ΠΟΥ ΔΟΥΛΕΥΟΥΝ" if fp else ""))
        for r in fp[:12]:
            print(f"      {r['id']:<6} [{r['set'][:22]:<22}] "
                  f"{r['question'][:52]}")
        if conv:
            print(f"  ΑΠΟ ΑΥΤΑ ΜΕ ΙΣΤΟΡΙΚΟ    {len(conv)}  <== θα διέκοπτε "
                  f"συνομιλίες που σήμερα δουλεύουν")

    if args.agent:
        only_a = sorted(r["id"] for r in rows if r["agent"] and not r["regex"])
        only_r = sorted(r["id"] for r in rows if r["regex"] and not r["agent"])
        print("\nΔΙΑΦΩΝΙΑ")
        print(f"  μόνο ο agent  {len(only_a)}  {only_a[:15]}")
        print(f"  μόνο το regex {len(only_r)}  {only_r[:15]}")
        print("\nΤΟ ΚΡΙΣΙΜΟ: ο agent κερδίζει ΜΟΝΟ αν το «μόνο ο agent» "
              "περιέχει\nπραγματικές ασάφειες ΚΑΙ δεν φέρνει ψευδώς θετικά.")
    print("=" * 78)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ασαφής ερώτηση πριν το retrieval: regex ή agent;")
    ap.add_argument("--agent", action="store_true",
                    help="τρέξε ΚΑΙ τον ανιχνευτή Gemini (~89 κλήσεις)")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
