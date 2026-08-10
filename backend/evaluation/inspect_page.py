"""Τι γράφει μια συγκεκριμένη σελίδα του corpus; — για post-mortem ερωτήσεων.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Όταν μια αλλαγή στο retrieval μετακινεί σελίδες (βλ. compare_pages_rerankers),
το «γιατί χειροτέρεψε η απάντηση» απαντιέται ΜΟΝΟ κοιτώντας το κείμενο που
μπήκε ή βγήκε. Χωρίς αυτό, μένεις με εικασίες για n=1 περιπτώσεις.

    docker compose exec backend python evaluation/inspect_page.py 1706.03178.pdf 2
    docker compose exec backend python evaluation/inspect_page.py 1706.03178.pdf 2 --grep gVisor
    docker compose exec backend python evaluation/inspect_page.py --grep gVisor
"""
import argparse
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_name", nargs="?", help="π.χ. 1706.03178.pdf")
    ap.add_argument("page", nargs="?", type=int)
    ap.add_argument("--grep", default=None,
                    help="τύπωσε μόνο γραμμές που το περιέχουν (case-insensitive)")
    ap.add_argument("--chars", type=int, default=2500)
    args = ap.parse_args()

    if args.file_name and args.page is not None:
        where = {"$and": [{"file_name": args.file_name}, {"page": args.page}]}
        got = ai_core.collection.get(where=where)
        if not got["ids"]:
            print(f"Καμία εγγραφή για {args.file_name}:{args.page}")
            return 1
        pairs = sorted(zip(got["ids"], got["documents"]))
        text = "\n".join(d for _i, d in pairs)
        print(f"=== {args.file_name} σελίδα {args.page} · "
              f"{len(pairs)} chunks · {len(text)} χαρακτήρες ===\n")
        if args.grep:
            low = args.grep.lower()
            hits = [ln for ln in text.splitlines() if low in ln.lower()]
            print(f"[{len(hits)} γραμμές με '{args.grep}']")
            for ln in hits:
                print(f"  > {ln.strip()}")
        else:
            print(text[:args.chars])
            if len(text) > args.chars:
                print(f"\n... (+{len(text)-args.chars} χαρακτήρες)")
        return 0

    if args.grep:
        # Πού αλλού εμφανίζεται ο όρος σε ΟΛΟ το corpus;
        got = ai_core.collection.get(include=["documents", "metadatas"])
        low = args.grep.lower()
        seen = {}
        for doc, meta in zip(got["documents"], got["metadatas"]):
            if low in (doc or "").lower():
                key = (meta.get("file_name"), meta.get("page"))
                seen[key] = seen.get(key, 0) + 1
        print(f"'{args.grep}' σε {len(seen)} σελίδες:")
        for (f, p), n in sorted(seen.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            print(f"  {f}:{p}   ({n} chunks)")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
