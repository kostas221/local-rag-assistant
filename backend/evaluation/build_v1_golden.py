"""Φτιάχνει golden set για το corpus του v1 (2 Berkeley papers) από το v2.

ΓΙΑΤΙ:
Το v1 (διπλωματική, commit 0b853ef) αξιολογείται σε 20 ερωτήσεις — 18 in-corpus
+ 2 out_of_corpus. Η προφανής ένσταση σε εξέταση είναι «λίγες ερωτήσεις», και
δεν έχει καλή απάντηση. Στο v2 υπάρχουν ΗΔΗ γραμμένες και ΕΠΑΛΗΘΕΥΜΕΝΕΣ
ερωτήσεις (verify_keywords: 0 σφάλματα) που απαντώνται ΑΠΟΚΛΕΙΣΤΙΚΑ από τα ίδια
δύο papers. Μεταφέρονται αυτούσιες: ΚΑΜΙΑ αλλαγή corpus, ΚΑΝΕΝΑ re-ingest.

ΤΟ ΚΡΙΤΗΡΙΟ ΕΝΤΑΞΗΣ (αυστηρό επίτηδες):
ΟΛΑ τα keywords της ερώτησης πρέπει να υπάρχουν σε σελίδα ΕΝΟΣ ΑΠΟ ΤΑ ΔΥΟ v1
papers. Αν έστω ένα keyword ζει μόνο σε τρίτο paper, η ερώτηση ΑΠΟΡΡΙΠΤΕΤΑΙ —
αλλιώς μεταφέρουμε ερώτηση που το v1 corpus ΔΕΝ μπορεί να απαντήσει και το MRR
της θα ήταν μηδέν από κατασκευή.

ΤΑ out_of_corpus ΜΠΑΙΝΟΥΝ ΟΛΑ: δεν εξαρτώνται από το corpus (κανένα paper δεν
μιλά για Bitcoin/quantum/GDPR/clinical trials/GPU tensor cores). Το v1 έχει 2,
το v2 έχει 5 -> +3 δωρεάν τεστ της άμυνας κατά της ψευδαίσθησης.

    docker compose exec backend python evaluation/build_v1_golden.py
    docker compose exec backend python evaluation/build_v1_golden.py --out evaluation/golden_v1_expanded.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/app")

HERE = "/app/evaluation"
SOURCES = [f"{HERE}/golden_set_50.jsonl", f"{HERE}/golden_multihop_new.jsonl"]
V1_PAPERS = {"EECS-2009-28.pdf", "1902.03383v1.pdf"}


def load_pages():
    import chromadb
    db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(name="ai_research_docs")
    got = col.get(include=["documents", "metadatas"])
    pages = {}
    for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        meta = meta or {}
        key = (str(meta.get("file_name", "?")), meta.get("page"))
        pages.setdefault(key, []).append(doc or "")
    return {k: "\n".join(v).lower() for k, v in pages.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/golden_v1_expanded.jsonl")
    args = ap.parse_args()

    pages = load_pages()

    def files_with(kw):
        return {f for (f, _p), t in pages.items() if kw.lower() in t}

    picked, rejected = [], []
    for src in SOURCES:
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                if t.get("category") == "out_of_corpus":
                    picked.append(t)      # ανεξάρτητο corpus -> πάντα μέσα
                    continue
                ok = all(files_with(k) & V1_PAPERS for k in t["keywords"])
                (picked if ok else rejected).append(t)

    seen, out = set(), []
    for t in picked:
        if t["id"] not in seen:
            seen.add(t["id"])
            out.append(t)

    inc = [t for t in out if t.get("category") != "out_of_corpus"]
    ooc = [t for t in out if t.get("category") == "out_of_corpus"]
    by_cat = {}
    for t in inc:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1

    print("ΤΟ v1 ΕΧΕΙ ΣΗΜΕΡΑ: 18 in-corpus + 2 out_of_corpus = 20")
    print(f"ΝΕΟ ΣΕΤ:           {len(inc)} in-corpus + {len(ooc)} out_of_corpus "
          f"= {len(out)}\n")
    print("ανά κατηγορία:")
    for c, n in sorted(by_cat.items()):
        print(f"  {c:<14} {n}")
    print(f"\nαπορρίφθηκαν {len(rejected)} (keywords ζουν σε άλλα papers)")

    with open(args.out, "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nΓράφτηκε: {args.out}")
    print("\nΕΠΟΜΕΝΟ — ΜΕΣΑ ΣΤΟ v1 (commit 0b853ef), με ΤΗ ΣΕΙΡΑ:")
    print("  1. verify_keywords.py  στο νέο σετ   (το v1 corpus είναι ΑΛΛΟ: 386 chunks)")
    print("  2. reranker -> MiniLM-L-12")
    print("  3. measure_gate_margin.py            (το 0.15 είναι sigmoid — ΑΛΛΗ ΚΛΙΜΑΚΑ)")
    print("  4. run_eval --retrieval-only")
    print("  5. judge run — ΜΙΑ ΦΟΡΑ, στο τελικό")
    return 0


if __name__ == "__main__":
    sys.exit(main())
