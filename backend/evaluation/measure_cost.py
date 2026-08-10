"""Πόσο κοστίζουν 1.000 ερωτήσεις; — το τρίτο σκέλος του cost/latency/quality.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το project μετρά latency και quality εξαντλητικά, και κόστος καθόλου. Πρόσφατη
έρευνα (arXiv 2511.09545) ζητά να μετριούνται τα τρία ΜΑΖΙ, γιατί κάθε βελτίωση
του ενός πληρώνεται από τα άλλα δύο. Και είναι η πρώτη ερώτηση κάθε εταιρίας.

ΓΙΑΤΙ ΤΟ ΝΟΥΜΕΡΟ ΕΙΝΑΙ ΑΣΥΝΗΘΙΣΤΑ ΚΑΛΟ ΕΔΩ:
Το embedding (bge-m3) και το reranking (cross-encoder) τρέχουν ΤΟΠΙΚΑ σε CPU.
Δεν χρεώνονται ανά κλήση — μόνο ο server, σταθερά. Το ΜΟΝΟ μεταβλητό κόστος
είναι η γέννηση. Ένα GPU-based demo δεν μπορεί να πει το ίδιο.

ΤΙ ΜΕΤΡΑΕΙ ΚΑΙ ΤΙ ΕΚΤΙΜΑ:
  ΜΕΤΡΑΕΙ  το μέγεθος του context που φτάνει στο Gemini — τρέχοντας το ΠΡΑΓΜΑΤΙΚΟ
           retrieval, δωρεάν, χωρίς καμία κλήση γέννησης.
  ΕΚΤΙΜΑ   τα tokens από τους χαρακτήρες (chars/ratio). Ο tokenizer του Gemini
           δεν είναι διαθέσιμος τοπικά. Το --ratio βαθμονομείται από ΜΙΑ πραγματική
           μέτρηση: το measure_e2e.py δίνει τα αληθινά promptTokenCount.
Οι τιμές είναι ΠΑΡΑΜΕΤΡΟΙ με defaults — ΕΠΑΛΗΘΕΥΣΕ τες πριν τις δημοσιεύσεις.

    docker compose exec backend python evaluation/measure_cost.py
    docker compose exec backend python evaluation/measure_cost.py --out-tokens 700 --server-eur 5
"""
import argparse
import asyncio
import json
import statistics
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

HERE = "/app/evaluation"

# ΤΙΜΕΣ ΑΝΑ 1 ΕΚΑΤΟΜΜΥΡΙΟ TOKENS (USD). Defaults για gemini-2.5-flash.
# ΑΛΛΑΖΟΥΝ — επαλήθευσε στο https://ai.google.dev/pricing πριν δημοσιεύσεις.
DEFAULT_IN_USD = 0.30
DEFAULT_OUT_USD = 2.50


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{HERE}/golden_set_50.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="χαρακτήρες ανά token (βαθμονόμησε από measure_e2e.py)")
    ap.add_argument("--overhead-tokens", type=int, default=700,
                    help="system prompt + persona + οδηγίες, εκτός context")
    ap.add_argument("--out-tokens", type=int, default=600,
                    help="μέσα tokens απάντησης (από measure_e2e.py)")
    ap.add_argument("--thinking-tokens", type=int, default=512,
                    help="THINKING_BUDGET — χρεώνονται ως έξοδος")
    ap.add_argument("--in-usd", type=float, default=DEFAULT_IN_USD)
    ap.add_argument("--out-usd", type=float, default=DEFAULT_OUT_USD)
    ap.add_argument("--usd-per-eur", type=float, default=1.08)
    ap.add_argument("--server-eur", type=float, default=5.0,
                    help="μηνιαίο κόστος VPS (CPU-only αρκεί)")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        tests = [json.loads(ln) for ln in f if ln.strip()]
    tests = [t for t in tests if t.get("category") != "out_of_corpus"]
    if args.limit:
        tests = tests[:args.limit]

    print(f"{len(tests)} ερωτήσεις · ΜΗΔΕΝ κλήσεις γέννησης (μόνο retrieval)\n")

    chars, pages = [], []
    for n, t in enumerate(tests, start=1):
        got = await ai_core.search_documents(t["question"])
        c = sum(len(txt) for txt, _m in got)
        chars.append(c)
        pages.append(len(got))
        print(f"  [{n}/{len(tests)}] {t.get('id', '?')}  {len(got)} σελ. "
              f"{c:,} χαρ.", end="\r", flush=True)

    print(" " * 60)
    ctx_tokens = statistics.mean(chars) / args.ratio
    in_tokens = ctx_tokens + args.overhead_tokens
    out_tokens = args.out_tokens + args.thinking_tokens

    in_usd = in_tokens / 1e6 * args.in_usd
    out_usd = out_tokens / 1e6 * args.out_usd
    per_q_usd = in_usd + out_usd
    per_q_eur = per_q_usd / args.usd_per_eur

    print("=" * 70)
    print("ΑΝΑ ΕΡΩΤΗΣΗ")
    print("=" * 70)
    print(f"  σελίδες context      {statistics.mean(pages):>10.1f}")
    print(f"  χαρακτήρες context   {statistics.mean(chars):>10,.0f}")
    print(f"  tokens εισόδου       {in_tokens:>10,.0f}   "
          f"(context {ctx_tokens:,.0f} + overhead {args.overhead_tokens})")
    print(f"  tokens εξόδου        {out_tokens:>10,.0f}   "
          f"(απάντηση {args.out_tokens} + thinking {args.thinking_tokens})")
    print(f"  κόστος               {per_q_usd:>10.5f} $  ({per_q_eur:.5f} €)")

    print("\n" + "=" * 70)
    print("ΑΝΑ 1.000 ΕΡΩΤΗΣΕΙΣ")
    print("=" * 70)
    print(f"  Gemini είσοδος       {in_usd * 1000:>10.2f} $")
    print(f"  Gemini έξοδος        {out_usd * 1000:>10.2f} $")
    print(f"  embeddings (bge-m3)  {0:>10.2f} $   <-- ΤΟΠΙΚΑ, CPU")
    print(f"  reranking (MiniLM)   {0:>10.2f} $   <-- ΤΟΠΙΚΑ, CPU")
    print(f"  BM25 / fusion        {0:>10.2f} $   <-- ΤΟΠΙΚΑ, CPU")
    print(f"  {'ΣΥΝΟΛΟ':<20} {per_q_usd * 1000:>10.2f} $  "
          f"({per_q_usd * 1000 / args.usd_per_eur:.2f} €)")

    month = args.server_eur + per_q_eur * 1000
    print("\n" + "=" * 70)
    print("ΠΡΑΓΜΑΤΙΚΟ ΜΗΝΙΑΙΟ ΚΟΣΤΟΣ (1.000 ερωτήσεις/μήνα)")
    print("=" * 70)
    print(f"  VPS (CPU-only)       {args.server_eur:>10.2f} €")
    print(f"  Gemini               {per_q_eur * 1000:>10.2f} €")
    print(f"  {'ΣΥΝΟΛΟ':<20} {month:>10.2f} €")

    share = 100 * (per_q_eur * 1000) / month if month else 0
    print(f"\n  Η γέννηση είναι το {share:.0f}% του κόστους· το υπόλοιπο είναι")
    print("  σταθερό και ΔΕΝ κλιμακώνεται με τη χρήση. Αν το embedding και το")
    print("  reranking γίνονταν με API, θα ήταν το τρίτο μεταβλητό κόστος —")
    print("  εδώ είναι μηδέν επειδή τρέχουν σε CPU.")

    print("\nΠΡΟΣΟΧΗ ΣΤΗΝ ΑΚΡΙΒΕΙΑ:")
    print(f"  · τα tokens ΕΚΤΙΜΩΝΤΑΙ ως χαρακτήρες/{args.ratio}. Βαθμονόμησε το")
    print("    --ratio από ένα πραγματικό promptTokenCount (measure_e2e.py).")
    print(f"  · οι τιμές ({args.in_usd}$/{args.out_usd}$ ανά 1M) είναι defaults —")
    print("    επαλήθευσέ τες πριν τις δημοσιεύσεις.")
    print("  · το free tier (~20 αιτήματα/ημέρα) καλύπτει τη χρήση επίδειξης")
    print("    με κόστος 0 €.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
