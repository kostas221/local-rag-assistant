"""Η πύλη με k chunks αντί για ένα — ΜΗΔΕΝ κλήσεις Gemini.

Η ΥΠΟΘΕΣΗ (δική σου): η διαρροή του o4 περνάει επειδή ΕΝΑ chunk έτυχε να ανέβει
πάνω από το κατώφλι — ίσως από μία τυχαία λέξη. Αν απαιτήσουμε τα ΔΥΟ κορυφαία
chunks να είναι πάνω από το κατώφλι, η τυχαία μονή επιτυχία κόβεται, ενώ μια
πραγματικά σχετική ερώτηση έχει ούτως ή άλλως πολλά καλά chunks.

ΠΩΣ ΜΕΤΡΙΕΤΑΙ ΧΩΡΙΣ ΚΟΣΤΟΣ: το granularity_probe.csv κατέγραψε τη ΜΕΤΑΦΡΑΣΗ κάθε
ερώτησης ανά συνθήκη. Η μετάφραση είναι το μόνο βήμα που θέλει δίκτυο. Την ξανα-
δίνουμε αυτούσια (monkeypatch του optimize_query) και ξανατρέχουμε ΜΟΝΟ την
ανάκτηση+reranking τοπικά, καταγράφοντας ΟΛΗ την ταξινομημένη λίστα score αντί για
το max. Ίδιο query -> ίδια ανάκτηση -> τα νούμερα είναι συγκρίσιμα με το CSV.

ΓΙΑΤΙ ΤΟ COVERAGE ΜΕΝΕΙ ΤΟ ΙΔΙΟ ΟΠΟΥ ΠΕΡΝΑΕΙ: η πύλη μόνο ΚΟΒΕΙ ή ΑΦΗΝΕΙ· δεν
αλλάζει την κατάταξη. Άρα όποια ερώτηση περνάει και με k=2 επιστρέφει ΑΚΡΙΒΩΣ ό,τι
επέστρεφε με k=1 -> κρατάμε το coverage του CSV. Όποια κόβεται πάει σε 0.

ΚΡΙΤΗΡΙΑ ΓΡΑΜΜΕΝΑ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ:
  Ο κανόνας k=2 ΓΙΝΕΤΑΙ ΔΕΚΤΟΣ μόνο αν, στη συνθήκη CM_pages, σβήνει τη διαρροή
  του o4 ΚΑΙ κρατάει το coverage >= 95.0% (δηλαδή δεν κόβει καμία in-corpus που
  περνούσε). Αν κόψει έστω μία in-corpus, το κόστος είναι πραγματικό και μπαίνει
  στο τραπέζι ρητά, δεν κρύβεται πίσω από τον μέσο όρο.
  ΕΛΕΓΧΟΣ ΕΓΚΥΡΟΤΗΤΑΣ: με k=1 τα νούμερα ΠΡΕΠΕΙ να αναπαράγουν το CSV.
"""
import probe_domain_glossary as P  # noqa: I001  πρώτο: κάνει sys.path.insert("/app")

import asyncio
import csv
import os

import chromadb

import ai_core

CSV_IN = "evaluation/runs/granularity_probe.csv"
GOLDEN = "evaluation/golden_test_domains.jsonl"
CSV_OUT = "evaluation/runs/gate_topk.csv"
PAPERS = ["evaluation/test_papers/cureus-0015-00000046486.pdf",
          "evaluation/test_papers/s41598-017-03833-3.pdf"]
GATE = ai_core.MIN_RERANK_SCORE
KS = (1, 2, 3)

_top: dict = {"scores": None}


def _install_full_spy():
    """Σαν τον spy του probe, αλλά κρατάει ΟΛΗ την ταξινομημένη λίστα, όχι το max."""
    orig = ai_core.reranker.predict

    def spy(pairs, **kw):
        s = orig(pairs, **kw)
        if _top["scores"] is None and len(s):
            _top["scores"] = sorted((float(x) for x in s), reverse=True)
        return s

    ai_core.reranker.predict = spy


async def main() -> int:
    ai_core._save_translation_cache = lambda: None
    ai_core.ENABLE_CORRECTIVE = False

    # --- ίδιο απομονωμένο store· ξαναχτίζεται αν το /tmp καθάρισε ---
    client = chromadb.PersistentClient(path=P.TEST_DB)
    col = client.get_or_create_collection(
        name="test_domains",
        embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    if col.count() == 0:
        print("Το test store είναι άδειο -> ingest ξανά")
        for i, path in enumerate(PAPERS):
            ai_core.ingest_pdf(path, os.path.basename(path), user_id=P.TEST_USER,
                               is_public=False, doc_id=i)
    print(f"test store: {col.count()} chunks · πύλη = {GATE}")
    ai_core._bump_corpus_version()
    _install_full_spy()

    tests = {t["id"]: t for t in P.load_golden(GOLDEN)}
    with open(CSV_IN, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    out_rows = []
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    for cond, crows in by_cond.items():
        print(f"\n===== {cond} =====")
        for r in crows:
            t = tests[r["id"]]
            tr = r["translation"]

            async def _fixed(_q, _tr=tr, **_kw):
                return _tr

            ai_core.optimize_query = _fixed
            _top["scores"] = None
            await ai_core.search_documents(t["question"], target_filenames=None,
                                           user_id=P.TEST_USER)
            sc = _top["scores"] or []
            above = sum(1 for s in sc if s >= GATE)
            csv_best = (r["best_logit"] or "").strip()
            rec = {"condition": cond, "id": r["id"],
                   "category": t.get("category", "in_corpus"),
                   "csv_best": csv_best,
                   "csv_coverage": r["coverage"],
                   "n_above_gate": above,
                   "top1": f"{sc[0]:+.2f}" if len(sc) > 0 else "",
                   "top2": f"{sc[1]:+.2f}" if len(sc) > 1 else "",
                   "top3": f"{sc[2]:+.2f}" if len(sc) > 2 else ""}
            out_rows.append(rec)
            ooc = t.get("category") == "out_of_corpus"
            tag = "ooc" if ooc else "in "
            print(f"  {r['id']:<4} {tag} πάνω από πύλη: {above:>2}  "
                  f"top1 {rec['top1']:>6} top2 {rec['top2']:>6} top3 {rec['top3']:>6}"
                  f"   (CSV best {csv_best})")

    # --- Αξιολόγηση των κανόνων k -------------------------------------------
    print("\n" + "#" * 74)
    print(f"ΚΑΝΟΝΑΣ ΠΥΛΗΣ: απαίτησε k chunks με score >= {GATE}")
    print("#" * 74)
    print(f"{'συνθήκη':<18} {'k':>2} {'cov':>7} {'ooc σιωπηλά':>13}   διαρροές")
    for cond in by_cond:
        recs = [r for r in out_rows if r["condition"] == cond]
        inc = [r for r in recs if r["category"] != "out_of_corpus"]
        ooc = [r for r in recs if r["category"] == "out_of_corpus"]
        for k in KS:
            covs = [float(r["csv_coverage"]) if r["n_above_gate"] >= k else 0.0
                    for r in inc if r["csv_coverage"] != ""]
            leaks = [r["id"] for r in ooc if r["n_above_gate"] >= k]
            mean = sum(covs) / len(covs) if covs else 0.0
            lost = [r["id"] for r in inc
                    if r["csv_coverage"] not in ("", "0.0") and r["n_above_gate"] < k]
            mark = "  <-- ΣΗΜΕΡΑ" if k == 1 else ""
            print(f"{cond:<18} {k:>2} {mean:6.1f}% {len(ooc) - len(leaks):>8}/{len(ooc)}"
                  f"   {', '.join(leaks) if leaks else '(καμία)'}"
                  + (f"   ΧΑΝΕΙ: {', '.join(lost)}" if lost else "") + mark)

    with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nΓράφτηκε: {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
