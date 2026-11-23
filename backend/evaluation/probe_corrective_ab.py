"""10ο τρέξιμο (18/11/2026): το ΠΑΛΙΟ και το ΝΕΟ prompt της δεύτερης αναζήτησης, 3 φορές το καθένα.

ΓΙΑΤΙ ΧΩΡΙΣΤΟ PROBE: η δεύτερη αναζήτηση (corrective) τρέχει ΜΟΝΟ στις ερωτήσεις που κόβει ο φύλακας,
και η αναδιατύπωση του 3.8 ΔΕΝ επαναλαμβάνεται (δεν δέχεται temperature). Ένα τρέξιμο του scoreboard
δίνει ΜΙΑ κλήρωση και την παγώνει· εδώ κάθε κομμένη ερώτηση παίρνει N ανεξάρτητες αναδιατυπώσεις ανά
prompt, άρα φαίνεται και η διασπορά. Όλα τα άλλα ΙΔΙΑ: ίδιο store (418 chunks, volume eval_tmp), ΙΔΙΕΣ
μεταφράσεις (από το αρχείο του τρεξίματος --run, όχι νέες κλήσεις), ίδια κατώφλια.

ΤΙ ΜΕΤΡΙΕΤΑΙ ΑΝΑ ΠΡΟΣΠΑΘΕΙΑ (κατώφλι CORRECTIVE_MIN_SCORE):
  ΔΙΑΡΡΟΕΣ ooc  : out_of_corpus του κύριου σετ που πέρασε — ΑΠΑΓΟΡΕΥΤΙΚΟ
  κοντινές      : ερώτηση του near_ooc που πέρασε (θα έπρεπε να σωπαίνει)
  σωσμένες      : απαντήσιμη που πέρασε ΚΑΙ έφερε υλικό (κάλυψη λέξεων > 0)
  ψευδαισθήσεις : απαντήσιμη που πέρασε ΧΩΡΙΣ υλικό (κάλυψη 0)

    docker compose ... run --rm --no-deps -e DATABASE_URL=sqlite:// -e GEMINI_MODEL=gemini-3.8-flash \\
        -v eval_tmp:/tmp backend python -u evaluation/probe_corrective_ab.py \\
        --run evaluation/runs/scoreboard/g38b.json --attempts 3 --csv evaluation/runs/corrective_ab.csv
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
from probe_corrective_attempts import _retrieve, coverage

import gemini_rest

HERE = "/app/evaluation"
SETS = {"main": "golden_set_50.jsonl", "hard_old": "golden_hard_paraphrase.jsonl",
        "hard_new": "golden_hard_new.jsonl", "near_ooc": "golden_near_ooc.jsonl"}
LOCK = "/tmp/scoreboard.lock"  # noqa: S108  ίδιο store με το scoreboard -> ίδια κλειδαριά

# Το prompt ΠΡΙΝ από το 10ο τρέξιμο, αυτούσιο (ai_core, 31/10/2026), για τη σύγκριση Α/Β.
OLD_PROMPT = (
    "The following search query returned no sufficiently relevant results from a "
    "corpus of scientific research papers on {domain}.\n\n"
    "Rewrite it as a keyword-rich search query using the standard technical "
    "terminology of that field. Replace vague or conversational wording with the "
    "terms a paper would actually use. Do NOT add facts, system names or numbers "
    "that are not implied by the original question. If the question is about a "
    "topic OUTSIDE that field, return it unchanged.\n\n"
    "Output ONLY the rewritten query. No quotes, no extra text.\n\n"
    "Original query: {query}\nRewritten query:"
)


def load_tests(run_path: str) -> list[dict]:
    with open(run_path, encoding="utf-8") as f:
        trans = {(r["set"], r["id"]): r.get("translation") for r in json.load(f)["rows"]}
    out = []
    for sk, name in SETS.items():
        with open(os.path.join(HERE, name), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = sk
                    t["_q"] = trans.get((sk, t["id"])) or t["question"]
                    out.append(t)
    return out


def kind(t: dict) -> str:
    if t["_set"] == "near_ooc":
        return "near"
    return "ooc" if t.get("category") == "out_of_corpus" else "answerable"


async def run(args) -> int:
    ai_core, _near = E.open_store(False)
    if ai_core.collection.count() != 418:
        print(f"!! Το store έχει {ai_core.collection.count()} chunks, όχι 418 — σταματάω")
        return 1
    tests = load_tests(args.run)
    thr = ai_core.CORRECTIVE_MIN_SCORE
    prompts = {"old": OLD_PROMPT, "new": ai_core._CORRECTIVE_PROMPT}
    allowed_ids = ai_core.collection.get(where=ai_core._build_where(None, None), include=[])["ids"]
    idx, dm = ai_core._get_bm25_index(), ai_core._get_dense_matrix()
    print(f"{len(tests)} ερωτήσεις · gate {ai_core.MIN_RERANK_SCORE} · corrective {thr} · "
          f"domain «{ai_core._CORPUS_DOMAIN}» · {args.attempts} προσπάθειες ανά prompt\n")

    rows = []
    for t in tests:
        ranked = await _retrieve(t["_q"], allowed_ids, idx, dm)
        if ranked[0][0] >= ai_core.MIN_RERANK_SCORE:
            continue                                  # ο φύλακας δεν κόβει -> δεν τρέχει corrective
        print(f"{t['_set']:<9}{t['id']:<6}[{kind(t):<10}] 1ο πέρασμα {ranked[0][0]:+6.2f}  «{t['_q'][:60]}»")
        for pname, tmpl in prompts.items():
            for a in range(1, args.attempts + 1):
                row = {"set": t["_set"], "id": t["id"], "kind": kind(t), "prompt": pname, "attempt": a,
                       "first_best": round(ranked[0][0], 3), "rewrite": "", "best": None, "cov": 0.0,
                       "passed": 0}
                try:
                    nq = (await gemini_rest.generate_once(
                        tmpl.format(query=t["_q"], domain=ai_core._CORPUS_DOMAIN),
                        model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)).strip(" \"'\n")
                except Exception as e:  # μία αποτυχία δεν ρίχνει το probe· καταγράφεται
                    row["rewrite"] = f"!! {type(e).__name__}"
                    rows.append(row)
                    print(f"    {pname} #{a} σφάλμα: {type(e).__name__}")
                    continue
                row["rewrite"] = nq
                if nq and nq.strip().lower() != t["_q"].strip().lower():   # όπως η παραγωγή
                    r2 = await _retrieve(nq, allowed_ids, idx, dm)
                    row["best"] = round(float(r2[0][0]), 3)
                    pages = await asyncio.to_thread(
                        ai_core._expand_to_pages, r2[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
                    cov = coverage("\n\n".join(p for p, _m in pages), t.get("keywords") or [])
                    row["cov"] = 0.0 if cov != cov else round(cov, 1)
                    row["passed"] = int(row["best"] >= thr)
                rows.append(row)
                score = "—" if row["best"] is None else f"{row['best']:+6.2f}"   # Python 3.11: χωρίς φωλιασμένα f-strings
                mark = " ΠΕΡΝΑ" if row["passed"] else "      "
                print(f"    {pname} #{a} {score:>7}{mark}  «{nq[:70]}»")

    print("\n" + "=" * 86)
    print(f"ΑΝΑ ΠΡΟΣΠΑΘΕΙΑ (κατώφλι {thr:+.2f}): ΔΙΑΡΡΟΕΣ ooc · κοντινές · σωσμένες · ψευδαισθήσεις")
    for pname in prompts:
        for a in range(1, args.attempts + 1):
            sel = [r for r in rows if r["prompt"] == pname and r["attempt"] == a and r["passed"]]
            leaks = sum(r["kind"] == "ooc" for r in sel)
            near = sum(r["kind"] == "near" for r in sel)
            saved = sum(r["kind"] == "answerable" and r["cov"] > 0 for r in sel)
            hall = sum(r["kind"] == "answerable" and r["cov"] == 0 for r in sel)
            names = " ".join(f"{r['id']}" for r in sel if r["kind"] != "answerable")
            print(f"  {pname} #{a}:  {leaks} · {near} · {saved} · {hall}" + (f"   (πέρασαν: {names})" if names else ""))
    errors = [r for r in rows if r["rewrite"].startswith("!!")]
    words = {p: [len(r["rewrite"].split()) for r in rows if r["prompt"] == p and not r["rewrite"].startswith("!!")]
             for p in prompts}
    for p, ws in words.items():
        if ws:
            print(f"  λέξεις αναδιατύπωσης {p}: διάμεσος {sorted(ws)[len(ws) // 2]} · μέσος {sum(ws) / len(ws):.1f}")
    print(f"  σφάλματα κλήσεων: {len(errors)}")
    print("=" * 86)

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Δεύτερη αναζήτηση: παλιό vs νέο prompt, N προσπάθειες")
    ap.add_argument("--run", required=True, help="runs/scoreboard/<label>.json — από εκεί οι μεταφράσεις")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard/probe στο ίδιο store (κλειδαριά {LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
