"""Ποιος ΙΣΧΥΡΙΣΜΟΣ ρίχνει μια απάντηση κάτω από το κατώφλι του NLI;

Το `probe_nli_entailment.py` δίνει ένα νούμερο ανά ερώτηση (`min_entail`) και
δεν λέει ΠΟΙΑ πρόταση το παρήγαγε. Χωρίς αυτό δεν ξέρεις αν το χαμηλό σκορ
είναι πραγματική αθεμελίωτη δήλωση ή τεχνούργημα του τεμαχισμού σε προτάσεις.

Δέχεται συγκεκριμένα id ώστε να μη ξαναγυρίσει ολόκληρο το σετ (2.16 s/ζεύγος).

    docker compose exec backend python evaluation/inspect_nli_claims.py h008 h012
"""
import argparse
import json
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

from probe_nli_entailment import (
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    entail_index,
    split_sentences,
    windows,
)


def main(args) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(args.source, encoding="utf-8") as fh:
        recs = json.load(fh)
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    want = {r["id"]: r for r in recs if r["id"] in set(args.ids)}
    missing = set(args.ids) - set(want)
    if missing:
        print(f"δεν βρέθηκαν: {sorted(missing)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    ent = entail_index(model)

    key_a = "answer_" + args.scenario
    key_p = "pages_" + args.scenario

    for qid in args.ids:
        r = want.get(qid)
        if not r:
            continue
        claims = split_sentences(r.get(key_a) or "")
        wins = windows((r.get(key_p) or [])[:args.top_pages],
                       args.window, args.stride)
        print("\n" + "=" * 78)
        print(f"{qid}  {len(claims)} ισχυρισμοί × {len(wins)} παράθυρα")
        print("=" * 78)
        scored = []
        for c in claims:
            best = 0.0
            for i in range(0, len(wins), args.batch):
                chunk = wins[i:i + args.batch]
                enc = tok(chunk, [c] * len(chunk), truncation=True,
                          max_length=args.max_len, padding=True,
                          return_tensors="pt")
                with torch.no_grad():
                    p = torch.softmax(model(**enc).logits, dim=-1)[:, ent]
                best = max(best, float(p.max()))
            scored.append((best, c))
        for s, c in sorted(scored):
            flag = "  <== ΤΟ ΧΑΜΗΛΟΤΕΡΟ" if (s, c) == min(scored) else ""
            print(f"  {s:.3f}  {c[:150]}{flag}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="per-claim entailment")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenario", choices=["A", "B"], default="B")
    ap.add_argument("--top-pages", type=int, default=3)
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=900)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    raise SystemExit(main(ap.parse_args()))
