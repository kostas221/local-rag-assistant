"""ΠΡΟΣΩΡΙΝΟ ΟΔΗΓΟΣ — ΣΕΛΙΔΕΣ vs CHUNKS στην εξαγωγή όρων.

ΔΕΝ ΑΛΛΑΖΕΙ ΤΙΠΟΤΑ. Εισάγει το probe_domain_glossary και το backfill_glossary και
τα χρησιμοποιεί ως έχουν· ό,τι πειράζεται (ai_core globals) πειράζεται ΣΤΗ ΜΝΗΜΗ
αυτής της διεργασίας, όπως ήδη κάνει το ίδιο το probe.

ΤΟ ΕΡΩΤΗΜΑ:
    Το CD (95.0%, ooc 5/5) μετρήθηκε περνώντας CHUNKS στο cleaned_terms
    (probe_domain_glossary.py:392 -> by_paper γεμίζει με documents, :400 -> cleaned_terms(docs)).
    Η ΠΑΡΑΓΩΓΗ περνάει ΣΕΛΙΔΕΣ (ai_core.ingest_pdf -> corpus_glossary.extract_glossary).
    Το κατώφλι boilerplate `d >= 0.9*n` σημαίνει άλλο πράγμα στα δύο: μετρήθηκε
    στα 7 cloud papers ότι η κοκκίωση σελίδας κόβει mapreduce/serverless/cloud/faas,
    η κοκκίωση chunk κανένα. Άρα η διαδρομή που τρέχει σε ΚΑΘΕ ΝΕΟ UPLOAD σήμερα
    δεν είναι αυτή που μετρήθηκε.

ΟΙ ΣΥΝΘΗΚΕΣ (μόνη μεταβλητή: η κοκκίωση της εξαγωγής όρων):
    C0_no_glossary  κενοί όροι                       -> πάτωμα, ίδιο με πριν
    CD_chunks       cleaned_terms(chunks)            -> CONTROL, αναπαράγει το CD
    CM_pages        document_terms(σελίδες)          -> Η ΠΑΡΑΓΩΓΗ ΣΗΜΕΡΑ
    Ίδιο domain, ίδια οδηγία (_G_CONSTRAINED) σε CD/CM.

ΚΡΙΤΗΡΙΑ ΓΡΑΜΜΕΝΑ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ:
  1. ΕΓΚΥΡΟΤΗΤΑ: το CD_chunks ΠΡΕΠΕΙ να βγει 95.0% με 5/5 σιωπηλές. Αλλιώς κάτι
     άλλο άλλαξε από το προηγούμενο τρέξιμο και ΟΛΟ το τρέξιμο είναι άκυρο.
  2. ΑΠΟΦΑΣΗ: αν CM_pages >= 95.0% ΚΑΙ 5/5 -> η κοκκίωση δεν είναι μοχλός, η
     παραγωγή μένει ως έχει και το εύρημα είναι καλλωπιστικό.
     Αν CM_pages < 95.0% Η διαρρεύσει -> η διαδρομή του ingest είναι υποβαθμισμένη
     και χρειάζεται διόρθωση (ΑΠΟ/ΣΕ στο ingest_pdf).
  3. ΘΟΡΥΒΟΣ: διαφορά < 1.0pp στο coverage ΔΕΝ μετράει ως διαφορά (n=20).

ΑΠΟΜΟΝΩΣΗ: /tmp/test_domains_chroma, χρήστης 999_999, _save_translation_cache
no-op. Το production chroma.sqlite3 και το translations.json δεν ανοίγουν.
"""
import asyncio
import csv
import os
import shutil

# Η ΣΕΙΡΑ ΜΕΤΡΑΕΙ: το probe_domain_glossary κάνει sys.path.insert(0, "/app") στη
# γραμμή 60 του, ΣΤΟ IMPORT TIME. Χωρίς αυτό πρώτο, τα modules του backend root
# (corpus_glossary, ai_core, backfill_glossary) δεν βρίσκονται.
import probe_domain_glossary as P  # noqa: I001  isort: skip

import chromadb  # noqa: E402
import corpus_glossary as cg  # noqa: E402

import ai_core  # noqa: E402
import backfill_glossary as bf  # noqa: E402

PAPERS = ["evaluation/test_papers/cureus-0015-00000046486.pdf",
          "evaluation/test_papers/s41598-017-03833-3.pdf"]
GOLDEN = "evaluation/golden_test_domains.jsonl"
CSV_OUT = "evaluation/runs/granularity_probe.csv"


async def main() -> int:
    ai_core._save_translation_cache = lambda: None      # μη μολύνεις την παραγωγή
    ai_core.ENABLE_CORRECTIVE = False                   # ντετερμινισμός, 1ο pass μόνο
    print("ENABLE_CORRECTIVE = False · translations.json = no-op")

    if os.path.exists(P.TEST_DB):
        shutil.rmtree(P.TEST_DB)
    client = chromadb.PersistentClient(path=P.TEST_DB)
    col = client.get_or_create_collection(
        name="test_domains",
        embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col

    print(f"Ingest {len(PAPERS)} papers στο {P.TEST_DB} ...")
    for i, path in enumerate(PAPERS):
        ok = ai_core.ingest_pdf(path, os.path.basename(path), user_id=P.TEST_USER,
                                is_public=False, doc_id=i)
        print(f"  {os.path.basename(path)}: {'OK' if ok else 'ΚΕΝΟ'}")
    ai_core._bump_corpus_version()
    P._install_reranker_spy()

    # --- CD: ΑΚΡΙΒΩΣ όπως το χτίζει το probe (γρ. 389-392, 400, 421) ---
    got = col.get(include=["documents", "metadatas"])
    by_paper: dict[str, list[str]] = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        by_paper.setdefault(meta["file_name"], []).append(doc)
    chunk_terms = ", ".join(dict.fromkeys(
        t for docs in by_paper.values() for t in P.cleaned_terms(docs)))

    # --- CM: ΑΚΡΙΒΩΣ ό,τι γράφει το ingest σήμερα (σελίδες) ---
    files = bf.collect_by_file(col)
    page_terms = ", ".join(dict.fromkeys(
        t for fn in by_paper for t in cg.document_terms(bf.page_texts_of(files[fn]))))

    for fn in by_paper:
        pages = bf.page_texts_of(files[fn])
        tc, tp = P.cleaned_terms(by_paper[fn]), cg.document_terms(pages)
        print(f"\n[{fn}]  {len(pages)} σελίδες / {len(by_paper[fn])} chunks")
        print(f"  CHUNKS : {', '.join(tc)}")
        print(f"  ΣΕΛΙΔΕΣ: {', '.join(tp)}")
        print(f"  -> χάνονται με σελίδες: {', '.join(w for w in tc if w not in tp) or '(τίποτα)'}")

    conds = [
        ("C0_no_glossary", P.GENERIC_DOMAIN, "", P._G_PROD),
        ("CD_chunks", P.GENERIC_DOMAIN, chunk_terms, P._G_CONSTRAINED),
        ("CM_pages", P.GENERIC_DOMAIN, page_terms, P._G_CONSTRAINED),
    ]

    tests = P.load_golden(GOLDEN)
    rows, results = [], []
    for name, dom, trm, tmpl in conds:
        mc, leaks = await P.run_condition(name, dom, trm, tmpl, tests, rows)
        results.append((name, mc, leaks))

    print("\n" + "#" * 70)
    print("ΣΥΝΟΨΗ — coverage (in-corpus) ΚΑΙ διαρροές (out_of_corpus, n=5)")
    n_ooc = sum(1 for t in tests if t.get("category") == "out_of_corpus")
    m0 = results[0][1]
    print(f"{'συνθήκη':<20} {'cov':>6} {'Δ vs C0':>9}   ooc σιωπηλά")
    for name, mc, leaks in results:
        line = f"{name:<20} {mc:5.1f}% {mc - m0:>+8.1f}   {n_ooc - len(leaks)}/{n_ooc}"
        print(line + (f"  <-- ΔΙΑΡΡΟΕΣ: {', '.join(leaks)}" if leaks else ""))

    cd = next(r for r in results if r[0] == "CD_chunks")
    cm = next(r for r in results if r[0] == "CM_pages")
    print("\nΕΛΕΓΧΟΣ ΕΓΚΥΡΟΤΗΤΑΣ (κριτήριο 1): CD_chunks = "
          f"{cd[1]:.1f}% / {n_ooc - len(cd[2])}/{n_ooc}  "
          + ("OK, αναπαράγει το 95.0%/5-5" if cd[1] >= 95.0 and not cd[2]
             else "ΑΠΟΤΥΧΙΑ -> το τρέξιμο είναι ΑΚΥΡΟ"))
    print(f"ΑΠΟΦΑΣΗ (κριτήριο 2): CM_pages = {cm[1]:.1f}% / {n_ooc - len(cm[2])}/{n_ooc}  "
          + ("-> η κοκκίωση ΔΕΝ είναι μοχλός, η παραγωγή μένει ως έχει"
             if cm[1] >= 95.0 and not cm[2]
             else "-> η διαδρομή του ingest ΕΙΝΑΙ υποβαθμισμένη"))

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nΓράφτηκε: {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
