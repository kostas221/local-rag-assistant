"""Φάση 0.2, βήμα 7: ΤΙ έφερε το σύστημα στις αποτυχίες του νέου hard set;

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Το eval_hard_new.py μέτρησε 7 «απάντησε από σελίδες χωρίς τις λέξεις-κλειδιά» και
    1 σιωπή. Πριν τα πούμε λάθη του ΣΥΣΤΗΜΑΤΟΣ, κοιτάμε με το μάτι μήπως φταίει το
    ΣΕΤ (κανόνας της Φάσης 0.1): η παράφραση έχασε το «ποιο paper» και την απαντά κι
    άλλο paper, ή οι λέξεις-κλειδιά που σφίξαμε 25/9 είναι πολύ στενές.

ΤΙ ΔΕΙΧΝΕΙ ανά ερώτηση:
    η ερώτηση · η γονική · η απάντηση-αναφορά · λέξεις-κλειδιά ΝΕΕΣ και ΠΑΛΙΕΣ
    κάθε σελίδα που γύρισε: ποιες λέξεις έχει + η πρόταση της σελίδας που μοιάζει
    περισσότερο με την απάντηση (λεξιλογική επικάλυψη — ΔΕΙΚΤΗΣ για το μάτι, όχι κριτής)
    οι σελίδες-στόχοι (όπου βρίσκονται οι νέες λέξεις) και αν ήρθαν
    Στη σιωπή: τι θα γύριζε ΧΩΡΙΣ φύλακα (κατώφλι -99), για να φανεί τι βρήκε.

ΜΗΔΕΝ κλήσεις Gemini: οι μεταφράσεις από το runs/hard_translations.json και η
αναδιατύπωση του corrective από το runs/hard_new_eval.csv (ίδια με του τρεξίματος).
Κάθε απρόσμενη κλήση Gemini ΣΤΑΜΑΤΑ το script — δεν αλλάζει σιωπηλά η ανάκτηση.
Έλεγχος αναπαραγωγής: το best1 πρέπει να βγει ίδιο με το CSV.

    docker compose exec backend python evaluation/inspect_hard_failures.py
    docker compose exec backend python evaluation/inspect_hard_failures.py --ids h109 h126
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_near_ooc as B
import eval_hard_new as H
import eval_near_ooc as E

import gemini_rest

EVAL_CSV = os.path.join(E.HERE, "runs", "hard_new_eval.csv")
OUT_JSON = os.path.join(E.HERE, "runs", "hard_failures_pages.json")
MULTIHOP = os.path.join(E.HERE, "golden_multihop_new.jsonl")
CORRECTIVE_PREFIX = "The following search query"


def best_sentence(text: str, ref: set[str]) -> str:
    """Η πρόταση της σελίδας με τις περισσότερες κοινές λέξεις με την απάντηση."""
    sents = re.split(r"(?<=[.!?])\s+|\n{2,}", " ".join(text.split()))
    scored = [(len(B.content_words(s) & ref), s) for s in sents if len(s) > 30]
    if not scored:
        return ""
    score, s = max(scored, key=lambda x: x[0])
    return f"({score} κοινές) {s[:280]}"


def corpus_pages(ai_core) -> dict:
    got = ai_core.collection.get(include=["documents", "metadatas"])
    pages: dict = {}
    for d, m in zip(got["documents"], got["metadatas"]):
        pages.setdefault((m.get("file_name"), m.get("page")), []).append(d or "")
    return {k: "\n".join(v).lower() for k, v in pages.items()}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    with open(EVAL_CSV, encoding="utf-8") as f:
        csv_rows = {r["id"]: r for r in csv.DictReader(f)}
    todo = [r for r in csv_rows.values()
            if r["set"] == "νέο" and (r["id"] in args.ids if args.ids else r["verdict"] != "με υλικό")]
    golden = {r["id"]: r for r in E.load_jsonl(H.NEW_PATH)}
    parents = {r["id"]: r for r in E.load_jsonl(E.GOLDEN_50) + E.load_jsonl(MULTIHOP)}

    ai_core, _near = E.open_store()
    with open(H.TRANS_PATH, encoding="utf-8") as f:
        ai_core._translation_cache.update(json.load(f))
    missing = [r["id"] for r in todo
               if golden[r["id"]]["lang"] == "el" and r["question"] not in ai_core._translation_cache]
    if missing:
        print(f"!! Λείπει σωσμένη μετάφραση για {missing} — θα χρειαζόταν Gemini, σταματάω")
        return 1

    fixed = {"rw": ""}

    async def no_gemini(prompt, **_kw):
        if prompt.startswith(CORRECTIVE_PREFIX) and fixed["rw"]:
            E._prompts.append((prompt[:60], fixed["rw"]))
            return fixed["rw"]
        raise RuntimeError(f"απρόσμενη κλήση Gemini: {prompt[:60]!r}")

    gemini_rest.generate_once = no_gemini
    allpages = corpus_pages(ai_core)
    gate = ai_core.MIN_RERANK_SCORE
    dump = []

    for r in todo:
        g = golden[r["id"]]
        p = parents.get(g["parent"], {})
        new_kw, old_kw = g["keywords"], g.get("keywords_parent", g["keywords"])
        ref = (B.content_words(g["reference_answer"]) | B.content_words(g.get("question_en", ""))
               | B.content_words(p.get("question", "")) | {k.lower() for k in new_kw + old_kw})

        fixed["rw"] = r["rewrite"] if r["outcome"] == "passed_corrective" else ""
        ai_core.ENABLE_CORRECTIVE = r["outcome"] == "passed_corrective"
        if r["outcome"] == "cut":
            ai_core.MIN_RERANK_SCORE = -99.0      # τι ΘΑ γύριζε χωρίς φύλακα
        pages, b1, _b2, _rw, _out = await E.retrieve(ai_core, r["question"])
        ai_core.MIN_RERANK_SCORE = gate

        same = E.fmt(b1) == r["best1"]
        print("\n" + "=" * 90)
        print(f"{r['id']} ({g['hard_type']}, {g['category']}) · ΑΠΟΦΑΣΗ: {r['verdict']} · "
              f"best1 {E.fmt(b1)} {'(ίδιο με το τρέξιμο)' if same else '!! ΔΙΑΦΟΡΕΤΙΚΟ από ' + r['best1']}")
        print(f"  ΕΡΩΤΗΣΗ     {g['question']}")
        if g.get("question_en") and g["lang"] == "el":
            print(f"  (αγγλικά)   {g['question_en']}")
        if r["translation"]:
            print(f"  μετάφραση   {r['translation']}")
        if r["outcome"] == "passed_corrective":
            print(f"  αναδιατύπ.  {r['rewrite']}")
        if r["outcome"] == "cut":
            print("  ** ΚΟΠΗΚΕ — παρακάτω ό,τι θα γύριζε ΧΩΡΙΣ φύλακα **")
        print(f"  ΓΟΝΙΚΗ {g['parent']}  {p.get('question', '?')}")
        print(f"  ΑΠΑΝΤΗΣΗ    {g['reference_answer']}")
        print(f"  λέξεις ΝΕΕΣ {new_kw}   ΠΑΛΙΕΣ {old_kw}")

        got = []
        for text, meta in pages:
            low = text.lower()
            f, pg = meta.get("file_name", "?"), meta.get("page")
            hit_new = [k for k in new_kw if k.lower() in low]
            hit_old = [k for k in old_kw if k.lower() in low] if old_kw != new_kw else hit_new
            got.append((f, pg))
            print(f"   - {f}:{pg}  νέες {hit_new or '—'}  παλιές {hit_old or '—'}")
            print(f"       {best_sentence(text, ref)}")
            dump.append({"id": r["id"], "file": f, "page": pg, "text": text})

        need = max(1, len(new_kw) - 1)
        targets = sorted(k for k, t in allpages.items()
                         if sum(kw.lower() in t for kw in new_kw) >= need)
        mark = ", ".join(f"{f}:{pg}{' ✓ΗΡΘΕ' if (f, pg) in got else ''}" for f, pg in targets)
        print(f"  ΣΕΛΙΔΕΣ-ΣΤΟΧΟΙ (≥{need} νέες λέξεις): {mark or 'καμία!'}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f"\nΠλήρες κείμενο σελίδων: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
