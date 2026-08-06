"""Δουλεύει καθόλου η αναδιατύπωση; — ΠΡΙΝ γραφτεί ο corrective agent.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το CLAUDE.md ζητά «όταν το gate κόβει, αναδιατύπωση και επανάληψη». Πριν μπει
αυτό στο hot path του search_documents, πρέπει να απαντηθούν δύο ερωτήσεις που
δεν έχουν προφανή απάντηση:
  1. ΣΩΖΕΙ; πόσες από τις κομμένες in-corpus περνούν το gate στο 2ο pass;
  2. ΔΙΑΡΡΕΕΙ; πόσα out_of_corpus περνούν επειδή τους δώσαμε δεύτερη ευκαιρία;

Το (2) είναι το μόνο σκληρό κριτήριο του project: τα 5 out_of_corpus ΠΡΕΠΕΙ να
μείνουν κομμένα 5/5. Το q048 (GDPR) είναι στο -2.69, μόλις 0.69 κάτω από το
κατώφλι — αν κάτι σπάσει, θα σπάσει εκεί.

ΤΙ ΚΑΝΕΙ (read-only — ΜΗΔΕΝ αλλαγή στο ai_core.py):
αναπαράγει το pipeline, βρίσκει ποιες κόβονται, τις ξαναγράφει με Gemini και
ξανατρέχει ΟΛΟΚΛΗΡΟ το retrieval με το νέο query. Τα out_of_corpus περνούν από
την ΙΔΙΑ διαδικασία ακόμα κι αν το πρώτο pass τα έκοψε — αλλιώς δεν μετριέται
η διαρροή.

ΚΟΣΤΟΣ: μία κλήση Gemini ανά κομμένη ερώτηση (~15 συνολικά). Το prompt είναι
παραμετρικό (--prompt-file) ώστε παραλλαγές να δοκιμάζονται χωρίς νέο κώδικα.

    docker compose exec backend python evaluation/probe_corrective_rewrite.py
    docker compose exec backend python evaluation/probe_corrective_rewrite.py --csv evaluation/runs/corrective_v1.csv
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
import gemini_rest
from evaluation.eval_engine import GOLDEN_CORPUS

DEFAULT_SETS = [
    "/app/evaluation/golden_hard_paraphrase.jsonl",
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
]

# Η οδηγία «αν το θέμα είναι εκτός πεδίου, άφησέ το ως έχει» ΔΕΝ είναι
# διακοσμητική: είναι η μοναδική άμυνα ώστε το rewrite να μη μεταμφιέσει ένα
# out_of_corpus ερώτημα σε ερώτημα του πεδίου. Χωρίς αυτήν, το «quantum
# computing performance» γίνεται «quantum computing cloud datacenter» και το
# gate δεν έχει τρόπο να το ξεχωρίσει.
DEFAULT_PROMPT = """\
The following search query returned no sufficiently relevant results from a \
corpus of computer-science papers on cloud computing, serverless computing and \
distributed systems.

Rewrite it as a keyword-rich search query using the standard technical \
terminology of that field. Replace vague or conversational wording with the \
terms a paper would actually use. Do NOT add facts, system names or numbers \
that are not implied by the original question. If the question is about a \
topic OUTSIDE that field, return it unchanged.

Output ONLY the rewritten query. No quotes, no extra text.

Original query: {query}
Rewritten query:"""


def load_sets(paths):
    tests = []
    seen = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    if t["id"] in seen:
                        continue
                    seen.add(t["id"])
                    t["_set"] = os.path.basename(p)
                    tests.append(t)
    return tests


def first_hit(ordered, kws):
    for r, (_s, text) in enumerate(ordered, 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


class Pipeline:
    """Το retrieval του search_documents μέχρι το gate, χωρίς το gate."""

    def __init__(self):
        where = ai_core._build_where(GOLDEN_CORPUS, None)
        self.allowed = ai_core.collection.get(where=where, include=[])["ids"]
        self.idx = ai_core._get_bm25_index()
        self.dm = ai_core._get_dense_matrix()

    def run(self, query: str):
        d_ids = ai_core._dense_exact_ids(
            self.dm, query, self.allowed,
            min(ai_core.DENSE_CANDIDATES, len(self.allowed)))
        s_ids = ai_core._bm25_sparse_ids(
            self.idx, query, self.allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(
            d_ids, s_ids, self.idx["ids"], self.idx["texts"], self.idx["metas"],
            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=self.idx["pos"])
        pairs = [[query, it[1]] for it in rrf]
        scores = ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)
        return sorted(zip((float(x) for x in scores), [it[1] for it in rrf]),
                      key=lambda x: -x[0])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--prompt-file", default=None,
                    help="αρχείο με εναλλακτικό prompt· πρέπει να έχει {query}")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    prompt_tpl = DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_tpl = f.read()
        print(f"prompt: {args.prompt_file}")
    if "{query}" not in prompt_tpl:
        print("ΣΦΑΛΜΑ: το prompt δεν περιέχει {query}")
        return 1

    tests = load_sets(args.sets)
    pipe = Pipeline()
    thr = ai_core.MIN_RERANK_SCORE
    print(f"{len(tests)} ερωτήσεις · κατώφλι {thr}\n")

    # --- PASS 1: ποιες κόβονται; ---
    todo = []
    for t in tests:
        q = await ai_core.optimize_query(t["question"])
        ordered = pipe.run(q)
        best = ordered[0][0]
        ooc = t.get("category") == "out_of_corpus"
        # Τα out_of_corpus μπαίνουν ΠΑΝΤΑ στο retry — εκεί μετριέται η διαρροή.
        if best < thr or ooc:
            todo.append((t, q, best, first_hit(ordered, t["keywords"])))

    n_ooc = sum(1 for t, *_ in todo if t.get("category") == "out_of_corpus")
    print(f"PASS 1: {len(todo) - n_ooc} in-corpus κομμένες + {n_ooc} out_of_corpus "
          f"-> {len(todo)} rewrites\n")

    # --- PASS 2: rewrite + πλήρες retrieval ---
    print(f"{'id':<6} {'cat':<14} {'best1':>7} {'best2':>7} {'Δ':>7} "
          f"{'έκβαση':<12} {'kw@':>4}")
    print("-" * 76)

    rows = []
    for t, q1, best1, hit1 in todo:
        try:
            q2 = (await gemini_rest.generate_once(
                prompt_tpl.format(query=q1), model=ai_core.GEMINI_MODEL,
                api_key=ai_core.GEMINI_API_KEY)).strip(' "\'\n')
        except Exception as e:
            print(f"{t['id']:<6} rewrite ΑΠΕΤΥΧΕ: {e}")
            continue
        if not q2:
            q2 = q1

        ordered = pipe.run(q2)
        best2 = ordered[0][0]
        hit2 = first_hit(ordered, t["keywords"])
        ooc = t.get("category") == "out_of_corpus"

        if ooc:
            outcome = "ΔΙΑΡΡΟΗ" if best2 >= thr else "κομμένο"
        elif best1 < thr <= best2:
            outcome = "ΣΩΘΗΚΕ"
        elif best1 < thr:
            outcome = "ακόμα κομμ."
        else:
            outcome = "-"

        rows.append(dict(id=t["id"], category=t.get("category", ""),
                         q1=q1, q2=q2, best1=best1, best2=best2,
                         kw1=hit1, kw2=hit2, outcome=outcome))
        print(f"{t['id']:<6} {t.get('category',''):<14} {best1:>7.2f} {best2:>7.2f} "
              f"{best2-best1:>7.2f} {outcome:<12} {(hit2 if hit2 else '-')!s:>4}",
              flush=True)

    saved = [r for r in rows if r["outcome"] == "ΣΩΘΗΚΕ"]
    stuck = [r for r in rows if r["outcome"] == "ακόμα κομμ."]
    leaks = [r for r in rows if r["outcome"] == "ΔΙΑΡΡΟΗ"]
    ooc_rows = [r for r in rows if r["category"] == "out_of_corpus"]

    print("\n" + "=" * 76)
    print("ΑΠΟΤΕΛΕΣΜΑ")
    print("=" * 76)
    print(f"  ΣΩΘΗΚΑΝ:       {len(saved)}/{len(saved)+len(stuck)}"
          + (f"  -> {', '.join(r['id'] for r in saved)}" if saved else ""))
    print(f"  ακόμα κομμένες: {len(stuck)}"
          + (f"  -> {', '.join(r['id'] for r in stuck)}" if stuck else ""))
    print(f"  out_of_corpus:  {len(ooc_rows)-len(leaks)}/{len(ooc_rows)} κομμένα "
          + ("— ΤΟ ΚΡΙΤΗΡΙΟ ΚΡΑΤΗΣΕ" if not leaks
             else f"— ΔΙΑΡΡΟΗ: {', '.join(r['id'] for r in leaks)}"))

    if leaks:
        print("\n  ΟΙ ΔΙΑΡΡΟΕΣ — τι έγραψε το rewrite:")
        for r in leaks:
            print(f"    {r['id']}: {r['best1']:.2f} -> {r['best2']:.2f}")
            print(f"       '{r['q1']}'")
            print(f"    -> '{r['q2']}'")

    if saved:
        print("\n  ΟΙ ΣΩΣΜΕΝΕΣ — τι άλλαξε:")
        for r in saved:
            print(f"    {r['id']}: {r['best1']:.2f} -> {r['best2']:.2f} "
                  f"(kw θέση {r['kw1'] or '-'} -> {r['kw2'] or '-'})")
            print(f"       '{r['q1']}'")
            print(f"    -> '{r['q2']}'")

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
