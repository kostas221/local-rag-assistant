"""Τι κάνει ο generator ΧΩΡΙΣ κατώφλι; Είναι σωστός ή ψευδαισθάνεται;

ΤΟ ΕΡΩΤΗΜΑ
----------
Όλη η άμυνα του συστήματος μετριέται στο `MIN_RERANK_SCORE`. Αλλά ο generator
έχει ΔΙΚΗ ΤΟΥ άμυνα, τον κανόνα 3 του system prompt:

    "If the answer is not in the text, clearly state that you cannot find the
     answer in the provided documents"

Αυτή ΔΕΝ έχει μετρηθεί ΠΟΤΕ απομονωμένη. Πάντα την προστάτευε το gate από
πάνω της, άρα δεν ξέρουμε αν κρατάει μόνη της. Και αυτό αλλάζει τα πάντα:

  · αν ΚΡΑΤΑΕΙ  -> το gate δεν είναι το μοναδικό σημείο απόφασης, και η
    επένδυση σε verification stage έχει νόημα (οι λανθασμένες αρνήσεις
    ανακτώνται χωρίς να ανοίγει η πόρτα)
  · αν ΔΕΝ κρατάει -> το gate δικαιώνεται οριστικά και η γραμμή κλείνει

ΔΥΟ ΣΕΝΑΡΙΑ, ΟΧΙ ΕΝΑ
--------------------
    Α  gate OFF          -> απαντά από τις σελίδες του 1ου περάσματος
                            (h008/h009 έχουν εκεί κάλυψη 0%)
    Β  corrective OFF    -> απαντά από τις σελίδες του 2ου περάσματος
                            (h008/h009 έχουν εκεί κάλυψη 100%)

Το Β είναι η πραγματική επιλογή σχεδίασης: εμπιστεύεσαι τον corrective agent
χωρίς κατώφλι. Το Α είναι το χειρότερο σενάριο και μετριέται ως όριο.

ΝΤΕΤΕΡΜΙΝΙΣΜΟΣ
--------------
Οι αναδιατυπώσεις ΔΕΝ ξαναπαράγονται — διαβάζονται από το `corrective_verdict.csv`,
όπως και στο `probe_quote_first.py`. Συγκρίσιμο με τα προηγούμενα τρεξίματα.

ΠΩΣ ΚΡΙΝΕΤΑΙ, ΧΩΡΙΣ ΚΡΙΤΗ
-------------------------
  · κάλυψη keywords στην ΑΠΑΝΤΗΣΗ (όχι στις σελίδες) -> απάντησε σωστά;
  · μοτίβα αυτο-άρνησης EL/EN                        -> αρνήθηκε μόνος του;
Και τα δύο ντετερμινιστικά. Η ίδια η απάντηση γράφεται στο CSV, γιατί το
«ψευδαίσθηση ή όχι» θέλει ανθρώπινο μάτι στο τέλος.

ΤΟ ΑΠΑΓΟΡΕΥΤΙΚΟ: έστω ΜΙΑ out_of_corpus που απαντιέται ουσιαστικά.

ΚΟΣΤΟΣ: 2 γεννήσεις × 11 ερωτήσεις = 22 κλήσεις, γύρω στα 0.12 $.

    docker compose exec backend python evaluation/probe_no_gate.py \\
        --csv evaluation/runs/no_gate.csv
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_hard_paraphrase.jsonl"]
DEFAULT_SOURCE = os.path.join(HERE, "runs", "corrective_verdict.csv")

# Μοτίβα αυτο-άρνησης. Ελληνικά ΚΑΙ αγγλικά, γιατί το σετ έχει και τα δύο και
# ο κανόνας 1 του prompt επιβάλλει τη γλώσσα της ερώτησης.
#
# ΠΡΩΤΗ ΕΚΔΟΧΗ ΗΤΑΝ ΠΟΛΥ ΣΤΕΝΗ και έβγαλε ΔΥΟ ψεύτικες διαρροές: το «there are
# **no reported** benchmark results» και το «do **not explicitly mention**» δεν
# έπιαναν, επειδή απαιτούσαν το `not` ακριβώς πριν το ρήμα. Επιτρέπονται πλέον
# ως 3 παρεμβαλλόμενες λέξεις. Οι out_of_corpus τυπώνονται ΠΑΝΤΑ αυτούσιες:
# με n=5 ο έλεγχος με το μάτι είναι εφικτός και η regex δεν είναι αξιόπιστη μόνη.
_W = r"(?:\w+\s+){0,3}"
REFUSAL = re.compile(
    r"δεν\s+" + _W + r"(?:βρ[ίι]σκ|βρ[έε]θηκ|περι[έε]χ|αναφ[έε]ρ|υπ[άα]ρχ|"
    r"παρ[έε]χ|μπορ|δ[ίι]ν|συζητ|εξετ|σχολι|προσδιορ|τεκμηρι|ανακτ)"
    r"|ουδεμ[ίι]α\s+αναφορ|καμ[ίι]α\s+" + _W + r"(?:αναφορ|πληροφορ|μνε[ίι]α)"
    r"|there\s+(?:is|are|was|were)\s+no\s"
    r"|(?:do|does|did)\s+not\s+" + _W + r"(?:contain|mention|provide|report|"
    r"discuss|specify|include|address|state|say|detail|present)"
    r"|(?:is|are|was|were)\s+not\s+" + _W + r"(?:contained|mentioned|provided|"
    r"reported|discussed|specified|included|addressed|present|available|found)"
    r"|\bno\s+" + _W + r"(?:information|answer|mention|reference|results?|data|"
    r"details?|evidence|benchmark|figures?)\b"
    r"|not\s+(?:explicitly\s+|specifically\s+|directly\s+)?(?:mentioned|discussed|"
    r"stated|specified|provided|reported|addressed|contained|available|present)"
    r"|cannot\s+" + _W + r"(?:find|determine|answer|provide|locate)"
    r"|unable\s+to\s+" + _W + r"(?:find|determine|answer|provide|locate)",
    re.I)


def load_sets():
    by_id = {}
    for name in SETS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = name
                    by_id[t["id"]] = t
    return by_id


def coverage(text: str, keywords) -> float:
    if not keywords:
        return 0.0
    low = (text or "").lower()
    return 100.0 * sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


def classify(ans: str, cov_ans: float, cov_pages: float, category: str) -> str:
    """Ντετερμινιστική ταξινόμηση. Το `ΨΕΥΔΑΙΣΘΗΣΗ;` θέλει έλεγχο με το μάτι."""
    refused = bool(REFUSAL.search(ans or ""))
    if refused and cov_ans == 0.0:
        return "ΑΡΝΗΣΗ"
    if category == "out_of_corpus":
        return "ΔΙΑΡΡΟΗ" if not refused else "ΑΡΝΗΣΗ"
    if cov_ans >= 50.0:
        return "ΣΩΣΤΗ"
    if cov_pages == 0.0:
        return "ΨΕΥΔΑΙΣΘΗΣΗ;"
    return "ΜΕΡΙΚΗ"


def summarize(rows) -> None:
    print("\n" + "=" * 78)
    for label, key in (("Α  gate OFF          ", "class_A"),
                       ("Β  corrective OFF    ", "class_B")):
        tally = {}
        for r in rows:
            tally[r[key]] = tally.get(r[key], 0) + 1
        leaks = tally.get("ΔΙΑΡΡΟΗ", 0)
        flag = "   <== ΑΠΑΓΟΡΕΥΤΙΚΟ" if leaks else ""
        print(f"{label}" + "  ".join(f"{k} {v}" for k, v in sorted(tally.items()))
              + flag)
    print("-" * 78)

    ooc = [r for r in rows if r["category"] == "out_of_corpus"]
    for sc in ("A", "B"):
        ref = sum(1 for r in ooc if r[f"class_{sc}"] == "ΑΡΝΗΣΗ")
        print(f"ΤΟ ΚΡΙΣΙΜΟ — out_of_corpus που ο generator αρνήθηκε ΜΟΝΟΣ ΤΟΥ "
              f"(σενάριο {sc}):  {ref}/{len(ooc)}")
    print("-" * 78)
    print("ανά ερώτηση (Α -> Β):")
    for r in rows:
        print(f"  {r['id']:<6} {r['class_A']:<13} -> {r['class_B']:<13}"
              f"  σελ {r['cov_pages_A']:5.1f} -> {r['cov_pages_B']:5.1f}"
              f"  απάντ {r['cov_answer_A']:5.1f} -> {r['cov_answer_B']:5.1f}")
    print("=" * 78)


async def answer_of(question: str, retrieved) -> str:
    out = ""
    async for chunk in ai_core.ask_ai(question, target_filenames=GOLDEN_CORPUS,
                                      precomputed=retrieved):
        if chunk.get("type") == "text":
            out += chunk.get("data", "")
    return out


async def second_pass(rewrite, allowed_ids, idx, dm):
    """Στάδια 2-7 πάνω στην ΑΠΟΘΗΚΕΥΜΕΝΗ αναδιατύπωση — ίδια με το quote_first."""
    dense_ids = await asyncio.to_thread(
        ai_core._dense_exact_ids, dm, rewrite, allowed_ids,
        min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = await asyncio.to_thread(
        ai_core._bm25_sparse_ids, idx, rewrite, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    pairs = [[rewrite, it[1]] for it in rrf]
    scores = await asyncio.to_thread(
        lambda: ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE))
    ranked = sorted(zip([float(s) for s in scores], [it[1] for it in rrf],
                        [it[2] for it in rrf]), key=lambda x: x[0], reverse=True)
    pages = await asyncio.to_thread(
        ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
    return ranked[0][0], pages


def reclassify(path: str):
    """Ξαναταξινομεί από αποθηκευμένες απαντήσεις. ΜΗΔΕΝ γεννήσεις.

    Υπάρχει επειδή τα μοτίβα άρνησης αποδείχθηκαν εύθραυστα: μια διόρθωση της
    regex δεν επιτρέπεται να κοστίζει 22 νέες κλήσεις — και δεν θα ήταν καν το
    ίδιο πείραμα, αφού η γέννηση δεν είναι ντετερμινιστική.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for d in data:
        kw = d.get("keywords") or []
        out = {"id": d["id"], "set": "", "category": d["category"], "best2": 0.0,
               "rewrite": d.get("rewrite", "")}
        for sc in ("A", "B"):
            ans = d.get("answer_" + sc) or ""
            covp = coverage("\n".join(d.get("pages_" + sc) or []), kw)
            cova = coverage(ans, kw)
            out[f"cov_pages_{sc}"] = round(covp, 1)
            out[f"cov_answer_{sc}"] = round(cova, 1)
            out[f"class_{sc}"] = classify(ans, cova, covp, d["category"])
            out[f"len_{sc}"] = len(ans)
            out[f"answer_{sc}"] = ans.replace("\n", " ")[:700]
        rows.append(out)
    return rows, data


async def main(args) -> int:
    if args.reclassify:
        rows, _data = reclassify(args.reclassify)
        print(f"ΞΑΝΑΤΑΞΙΝΟΜΗΣΗ από {os.path.basename(args.reclassify)} "
              f"— ΜΗΔΕΝ νέες γεννήσεις\n")
        summarize(rows)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"→ γράφτηκε {args.csv}")
        return 0

    if not os.path.exists(args.source):
        print(f"Λείπει το {args.source} — τρέξε πρώτα probe_corrective_verdict.py")
        return 1
    with open(args.source, encoding="utf-8") as fh:
        source = list(csv.DictReader(fh))
    by_id = load_sets()

    print(f"{len(source)} κομμένες ερωτήσεις · gate={ai_core.MIN_RERANK_SCORE:+.2f} "
          f"corrective={ai_core.CORRECTIVE_MIN_SCORE:+.2f}")
    print("ΣΕΝΑΡΙΟ Α: gate OFF (1ο πέρασμα) · "
          "ΣΕΝΑΡΙΟ Β: corrective OFF (2ο πέρασμα)\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    # Το gate κάτω από κάθε πιθανό logit + ο agent κλειστός: το search_documents
    # επιστρέφει ΠΑΝΤΑ τις σελίδες του 1ου περάσματος.
    saved_gate, saved_corr = ai_core.MIN_RERANK_SCORE, ai_core.ENABLE_CORRECTIVE
    rows, dump = [], []
    try:
        for src in source:
            qid = src["id"]
            t = by_id.get(qid)
            if t is None:
                continue
            cat = src.get("category", "")
            kw = t.get("keywords") or []
            rewrite = (src.get("rewrite") or "").strip()
            print(f"{qid:<6} [{cat:<14}] {t['question'][:52]}...", flush=True)

            # ---------------------------------------------- Α: χωρίς gate
            ai_core.MIN_RERANK_SCORE = -999.0
            ai_core.ENABLE_CORRECTIVE = False
            retrieved = await ai_core.search_documents(
                t["question"], target_filenames=GOLDEN_CORPUS)
            covp_a = coverage("\n".join(x for x, _m in retrieved), kw)
            ans_a = await answer_of(t["question"], retrieved) if retrieved else ""
            cova_a = coverage(ans_a, kw)
            cls_a = classify(ans_a, cova_a, covp_a, cat)
            print(f"       Α  σελ {covp_a:5.1f}  απάντ {cova_a:5.1f}  {cls_a}")

            # ------------------------------- Β: 2ο πέρασμα χωρίς κατώφλι
            best2, pages = await second_pass(rewrite, allowed_ids, idx, dm)
            covp_b = coverage("\n".join(x for x, _m in pages), kw)
            ans_b = await answer_of(t["question"], pages)
            cova_b = coverage(ans_b, kw)
            cls_b = classify(ans_b, cova_b, covp_b, cat)
            print(f"       Β  σελ {covp_b:5.1f}  απάντ {cova_b:5.1f}  {cls_b}"
                  f"   (best2 {best2:+.2f})")

            rows.append({
                "id": qid, "set": src.get("set", ""), "category": cat,
                "best2": round(best2, 3),
                "cov_pages_A": round(covp_a, 1), "cov_answer_A": round(cova_a, 1),
                "class_A": cls_a, "len_A": len(ans_a),
                "cov_pages_B": round(covp_b, 1), "cov_answer_B": round(cova_b, 1),
                "class_B": cls_b, "len_B": len(ans_b),
                "rewrite": rewrite,
                "answer_A": (ans_a or "").replace("\n", " ")[:700],
                "answer_B": (ans_b or "").replace("\n", " ")[:700]})
            # Πλήρη δεδομένα για το `probe_nli_entailment.py`: εκείνο τρέχει
            # ΜΟΝΟ το NLI μοντέλο πάνω σε αυτά, με ΜΗΔΕΝ κλήσεις API.
            dump.append({
                "id": qid, "category": cat, "question": t["question"],
                "keywords": kw, "rewrite": rewrite,
                "answer_A": ans_a, "answer_B": ans_b,
                "pages_A": [x for x, _m in retrieved],
                "pages_B": [x for x, _m in pages]})
            if args.delay:
                await asyncio.sleep(args.delay)
    finally:
        ai_core.MIN_RERANK_SCORE, ai_core.ENABLE_CORRECTIVE = saved_gate, saved_corr

    if not rows:
        print("Καμία γραμμή.")
        return 1
    summarize(rows)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
        # Δίπλα στο CSV, ΠΛΗΡΗ δεδομένα για το NLI probe (0 κλήσεις εκεί).
        jpath = os.path.splitext(args.csv)[0] + "_full.json"
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, ensure_ascii=False, indent=1)
        print(f"→ γράφτηκε {jpath}  (είσοδος του probe_nli_entailment.py)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Χωρίς κατώφλι: σωστή απάντηση ή ψευδαίσθηση;")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--reclassify", default=None,
                    help="ξαναταξινομεί από no_gate_full.json, ΜΗΔΕΝ γεννήσεις")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
