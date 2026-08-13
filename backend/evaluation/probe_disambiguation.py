"""Αποσαφήνιση ΜΕΣΩ ανάκτησης: συμπλήρωσε πρώτα το υποκείμενο, ψάξε μετά.

ΓΙΑΤΙ ΔΕΝ ΕΙΝΑΙ ΤΑ ΗΔΗ ΑΠΟΡΡΙΦΘΕΝΤΑ
Κριτής ΝΑΙ/ΟΧΙ, quote-first και RBO συγκλίνουν στο ΙΔΙΟ 3/1/0 επειδή και οι
τρεις κρίνουν τις ΣΕΛΙΔΕΣ: ρωτούν «απαντούν αυτές την ερώτηση;» για μια ερώτηση
που δεν λέει σε τι αναφέρεται. Το h012 έδειξε ότι αυτό είναι κλειστό ΑΠΟ
ΚΑΤΑΣΚΕΥΗ — κάθε πρόταση του σωστού ΣΧΗΜΑΤΟΣ ικανοποιεί τυπικά μια ερώτηση με
ασυμπλήρωτο αναφορικό. Εδώ η σειρά αντιστρέφεται: πρώτα ονομάζουμε τα πιθανά
αναφερόμενα, μετά κάνουμε ανάκτηση για καθένα ξεχωριστά.

ΤΙ ΜΕΤΡΑΕΙ (read-only — ΜΗΔΕΝ αλλαγή στο ai_core.py)
  1. control ΔΩΡΕΑΝ: πού πυροδοτεί το _has_dangling_referent σε ΟΛΑ τα σετ.
     Αν σημαίνει έξω από το hard set, το «μηδέν κόστος στο happy path» πέφτει
     ΕΔΩ, πριν ξοδευτεί μία κλήση.
  2. μία κλήση Gemini ανά σημαία: «έως 3 αυτόνομες εκδοχές, μία ανά πιθανό
     αναφερόμενο», με τα ΠΡΑΓΜΑΤΙΚΑ papers του corpus μπροστά της.
  3. πλήρες 2ο πέρασμα (βήματα 2-7) ανά εκδοχή, με το ΚΑΝΟΝΙΚΟ gate -2.6 —
     όχι το corrective: οι εκδοχές είναι πλέον καλοδιατυπωμένες ερωτήσεις,
     άρα κάθονται στην κλίμακα για την οποία εκπαιδεύτηκε το ms-marco.
  4. κάλυψη keywords στην ΕΝΩΣΗ των σελίδων όσων εκδοχών περνούν.

ΤΟ ΣΚΕΛΟΣ ΑΣΦΑΛΕΙΑΣ (--ooc, προεπιλογή ΝΑΙ)
Ο μηχανισμός τρέχει ΑΝΑΓΚΑΣΤΙΚΑ και στις 5 out_of_corpus, παρόλο που ο
ανιχνευτής δεν σημαίνει ποτέ σε αυτές. Ερώτημα: αν κάποτε σημάνει, θα
κατασκευάσει «εύλογη» ερώτηση εντός corpus από το «πόσο κάνει το Bitcoin;»
και θα περάσει το gate; Αυτό είναι το μόνο σενάριο όπου ο μηχανισμός ΧΑΛΑΕΙ
κάτι που σήμερα δουλεύει.

ΕΤΥΜΗΓΟΡΙΑ ΑΝΑ ΕΡΩΤΗΣΗ
  καμία εκδοχή πάνω από το gate      -> ΣΙΩΠΗ ΠΑΡΑΜΕΝΕΙ (μηδέν κέρδος, μηδέν ζημιά)
  περνάει & κάλυψη > 0               -> ΛΥΝΕΤΑΙ
  περνάει & κάλυψη = 0               -> ΕΥΛΟΓΟ ΑΛΛΑ ΛΑΘΟΣ ΥΛΙΚΟ (το ρίσκο)
  out_of_corpus & περνάει            -> ΔΙΑΡΡΟΗ -> κλείνει οριστικά ο δρόμος

ΠΡΟΒΛΕΨΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (η καταγραφή έχει ΔΥΟ λάθος προβλέψεις σε δύο probes,
γι' αυτό γράφεται ΠΡΙΝ): h012 και h016 έχουν από ένα εύλογο αναφερόμενο ->
λύνονται· h009 πολλαπλά -> καλύπτεται από την απαρίθμηση· καμία διαρροή ooc.

ΚΟΣΤΟΣ: 1 κλήση Gemini ανά σημαία (~4) + 5 του σκέλους ασφαλείας = ~9.
Καμία κλήση γέννησης — μετράμε ΑΝΑΚΤΗΣΗ, όχι απάντηση.

    docker compose exec backend python evaluation/probe_disambiguation.py \
        --csv evaluation/runs/disambiguation.csv
    docker compose exec backend python evaluation/probe_disambiguation.py --scan-only
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
import gemini_rest

SETS = ["/app/evaluation/golden_hard_paraphrase.jsonl",
        "/app/evaluation/golden_set_50.jsonl",
        "/app/evaluation/golden_multihop_new.jsonl",
        "/app/evaluation/golden_conversations.jsonl",
        "/app/evaluation/golden_tables.jsonl"]

_PROMPT = (
    "A user asked a question about a corpus of computer-science papers, but the "
    "question refers to an entity it never names, so it cannot be answered as "
    "written.\n\n"
    "The corpus contains exactly these papers:\n{titles}\n\n"
    "Give up to {n} possible complete, self-contained versions of the question - "
    "one per plausible referent. Each version must NAME the paper, system or "
    "technique it refers to, and must be answerable from the corpus above. If "
    "only one referent is plausible, give only one version. Do NOT invent papers "
    "or systems that are not in the list.\n\n"
    "Output ONLY the versions, one per line. No numbering, no quotes, no other "
    "text.\n\n"
    "Question: {q}"
)


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def question_of(t: dict) -> str:
    """Τα golden_conversations έχουν followup/turn1_question, όχι question.

    Το verify_keywords.py ΕΣΚΑΓΕ ακριβώς εδώ (KeyError) και γι' αυτό το σετ δεν
    είχε ελεγχθεί ποτέ. Ίδιο μοτίβο, ίδια λύση.
    """
    return (t.get("question") or t.get("followup")
            or t.get("turn1_question") or "")


def load_golden() -> dict:
    out = {}
    for p in SETS:
        if not os.path.exists(p):
            print("ΛΕΙΠΕΙ: %s" % p)
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = os.path.basename(p)
                    out.setdefault(t["id"], t)
    return out


def corpus_titles(idx: dict, n_chars: int = 130) -> str:
    """Τα ΠΡΑΓΜΑΤΙΚΑ papers, με την πρώτη γραμμή της πρώτης τους σελίδας.

    Χωρίς αυτό το μοντέλο θα εφεύρισκε αναφερόμενα — και η αποσαφήνιση θα
    μετρούσε τη φαντασία του, όχι το corpus.
    """
    first = {}
    for txt, m in zip(idx["texts"], idx["metas"]):
        f, p = m.get("file_name", "?"), int(m.get("page", 10**6))
        if f not in first or p < first[f][0]:
            first[f] = (p, " ".join(txt.split())[:n_chars])
    return "\n".join("- %s: %s" % (f, s) for f, (_p, s) in sorted(first.items()))


async def versions_of(question: str, titles: str, n: int) -> list:
    raw = await gemini_rest.generate_once(
        _PROMPT.format(titles=titles, n=n, q=question),
        model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY,
        max_output_tokens=512)
    out = []
    for line in (raw or "").splitlines():
        s = line.strip().lstrip("-*0123456789.) ").strip(' "\'')
        if len(s) > 15 and s not in out:
            out.append(s)
    return out[:n]


def retrieve(q: str):
    """Βήματα 2-7 του search_documents. Ίδιο με το probe_corrective_floor."""
    where = ai_core._build_where(None, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    d_ids = ai_core._dense_exact_ids(
        dm, q, allowed, min(ai_core.DENSE_CANDIDATES, len(allowed)))
    s_ids = ai_core._bm25_sparse_ids(idx, q, allowed, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"], idx["metas"],
                            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    scores = ai_core.reranker.predict(
        [[q, it[1]] for it in rrf], batch_size=ai_core.RERANK_BATCH_SIZE)
    final = sorted(zip((float(x) for x in scores),
                       [it[1] for it in rrf], [it[2] for it in rrf]),
                   key=lambda x: x[0], reverse=True)
    pages = ai_core._expand_to_pages(final[:ai_core.EXPAND_INPUT],
                                     ai_core.MAX_PAGES)
    return final[0][0], pages


def coverage(pages, keywords) -> tuple:
    if not keywords:
        return 0.0, []
    blob = fold(" ".join(txt for txt, _m in pages))
    found = [k for k in keywords if fold(k) in blob]
    return 100.0 * len(found) / len(keywords), found


def page_key(m) -> str:
    return "%s:%s" % (m.get("file_name"), m.get("page"))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-versions", type=int, default=3)
    ap.add_argument("--scan-only", action="store_true",
                    help="μόνο ο ανιχνευτής — ΜΗΔΕΝ κλήση Gemini")
    ap.add_argument("--ooc", dest="ooc", action="store_true", default=True)
    ap.add_argument("--no-ooc", dest="ooc", action="store_false")
    ap.add_argument("--ids", default=None,
                    help="παράκαμψη του ανιχνευτή, π.χ. h012,h016")
    args = ap.parse_args()

    golden = load_golden()
    idx = ai_core._get_bm25_index()

    # --- 1. CONTROL: πού πυροδοτεί ο ανιχνευτής; ΔΩΡΕΑΝ -----------------
    flags, per_set = [], {}
    for qid, t in golden.items():
        q = question_of(t)
        per_set.setdefault(t["_set"], [0, 0])
        per_set[t["_set"]][1] += 1
        if q and ai_core._has_dangling_referent(q):
            flags.append(qid)
            per_set[t["_set"]][0] += 1

    print("=" * 78)
    print("ΑΝΙΧΝΕΥΤΗΣ ΑΝΑΦΟΡΙΚΩΝ — σημαίες ανά σετ (%d ερωτήσεις συνολικά)"
          % len(golden))
    for s, (hit, tot) in sorted(per_set.items()):
        print("  %-32s %d / %d" % (s, hit, tot))
    print("  σημαίες: %s" % (", ".join(sorted(flags)) or "-"))
    print("\nΤο «μηδέν κόστος στο happy path» ισχύει ΜΟΝΟ αν οι σημαίες είναι\n"
          "όλες στο hard set. Οτιδήποτε άλλο πληρώνει +1 κλήση Gemini σε\n"
          "ερώτηση που το σύστημα ήδη απαντάει.")
    if args.scan_only:
        return 0

    targets = ([i.strip() for i in args.ids.split(",")] if args.ids
               else sorted(flags))
    if args.ooc:
        ooc = [qid for qid, t in golden.items()
               if t.get("category") == "out_of_corpus"]
        targets += [q for q in sorted(ooc) if q not in targets]

    titles = corpus_titles(idx)
    print("\n%d papers στο corpus · %d ερωτήσεις · %d κλήσεις Gemini\n"
          % (titles.count("\n") + 1, len(targets), len(targets)))

    rows, verdicts = [], {}
    for qid in targets:
        t = golden.get(qid)
        if not t:
            print("%-6s ΛΕΙΠΕΙ από τα golden sets — παραλείπεται" % qid)
            continue
        q, kws = question_of(t), t.get("keywords", [])
        is_ooc = t.get("category") == "out_of_corpus"

        base_q = await ai_core.optimize_query(q)
        base_best, base_pages = retrieve(base_q)
        base_cov, _f = coverage(base_pages, kws)

        print("=" * 78)
        print("%s  [%s]%s" % (qid, t.get("hard_type", t.get("category", "")),
                              "  <-- ΣΚΕΛΟΣ ΑΣΦΑΛΕΙΑΣ" if is_ooc else ""))
        print("  ερώτηση : %s" % q)
        print("  ΣΗΜΕΡΑ  : best %+.2f  (gate %.1f -> %s)  κάλυψη %.0f%%"
              % (base_best, ai_core.MIN_RERANK_SCORE,
                 "ΚΟΒΕΙ" if base_best < ai_core.MIN_RERANK_SCORE else "ΠΕΡΝΑΕΙ",
                 base_cov))

        try:
            vs = await versions_of(q, titles, args.max_versions)
        except Exception as e:
            print("  ΑΠΕΤΥΧΕ η αποσαφήνιση: %s" % e)
            continue
        if not vs:
            print("  καμία εκδοχή — το μοντέλο δεν βρήκε πιθανό αναφερόμενο")

        union, passed = {}, 0
        for k, v in enumerate(vs, 1):
            best, pages = retrieve(v)
            cov, found = coverage(pages, kws)
            ok = best >= ai_core.MIN_RERANK_SCORE
            passed += ok
            if ok:
                for txt, m in pages:
                    union.setdefault(page_key(m), (txt, m))
            print("  v%d %s best %+6.2f  κάλυψη %5.1f%%  %s"
                  % (k, "ΠΕΡΝΑΕΙ" if ok else "κόβεται ", best, cov, v))
            print("       σελίδες: %s" % ", ".join(page_key(m)
                                                   for _x, m in pages))
            rows.append(dict(id=qid, set=t["_set"], ooc=is_ooc, version=k,
                             text=v, best=round(best, 4), passes=ok,
                             coverage=round(cov, 1),
                             keywords_found=";".join(found),
                             pages=";".join(page_key(m) for _x, m in pages)))

        u_pages = list(union.values())
        u_cov, u_found = coverage(u_pages, kws)
        if is_ooc:
            verdict = ("ΔΙΑΡΡΟΗ — κατασκεύασε ερώτηση εντός corpus" if passed
                       else "σιωπηλό (καμία εκδοχή πάνω από το gate)")
        elif not passed:
            verdict = "ΣΙΩΠΗ ΠΑΡΑΜΕΝΕΙ"
        elif u_cov > 0:
            verdict = "ΛΥΝΕΤΑΙ"
        else:
            verdict = "ΕΥΛΟΓΟ ΑΛΛΑ ΛΑΘΟΣ ΥΛΙΚΟ"
        verdicts[qid] = (verdict, passed, len(vs), u_cov, len(u_pages))
        print("  ΕΝΩΣΗ   : %d σελίδες (cap %d)  κάλυψη %.0f%%  %s"
              % (len(u_pages), ai_core.MAX_PAGES, u_cov,
                 ", ".join(u_found) or "-"))
        print("  ->  %s" % verdict)

    print("=" * 78)
    print("ΣΥΝΟΨΗ")
    for qid, (v, passed, n, u_cov, n_pg) in verdicts.items():
        print("  %-6s εκδοχές %d/%d πάνω από το gate · ένωση %d σελ · "
              "κάλυψη %5.1f%%  ->  %s" % (qid, passed, n, n_pg, u_cov, v))
    leaks = [q for q, (v, *_r) in verdicts.items() if v.startswith("ΔΙΑΡΡΟΗ")]
    solved = [q for q, (v, *_r) in verdicts.items() if v == "ΛΥΝΕΤΑΙ"]
    print("\n  ΛΥΝΟΝΤΑΙ: %s" % (", ".join(solved) or "καμία"))
    print("  ΔΙΑΡΡΟΕΣ: %s" % (", ".join(leaks) or "καμία"))
    print("\n  Το κριτήριο υιοθέτησης: ΜΗΔΕΝ διαρροές ΚΑΙ ≥1 λυμένη. Οτιδήποτε\n"
          "  άλλο κλείνει τον δρόμο — με μέτρηση αυτή τη φορά, όχι με εικασία.")

    if args.csv and rows:
        path = args.csv if os.path.isabs(args.csv) else "/app/" + args.csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
