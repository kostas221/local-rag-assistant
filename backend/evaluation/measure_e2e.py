"""Πού πάει ΠΡΑΓΜΑΤΙΚΑ ο χρόνος του χρήστη: retrieval vs γέννηση.

ΓΙΑΤΙ: μετά τα thread/batch fixes το warm retrieval έπεσε στα ~450ms. Αν η
γέννηση είναι 3-5s, τότε κάθε επόμενο ms που κερδίζουμε στο retrieval είναι
θόρυβος για τον χρήστη και ψάχνουμε σε λάθος μέρος. Πρώτα μετράμε, μετά
αποφασίζουμε πού αξίζει η επόμενη προσπάθεια.

ΤΙ ΔΕΙΧΝΕΙ ΕΠΙΠΛΕΟΝ: τα THINKING TOKENS. Το gemini-2.5-flash είναι thinking
model — τα tokens που "σκέφτεται" χρεώνονται στο budget ΚΑΙ στον χρόνο, αλλά
δεν φαίνονται στην απάντηση. Αν είναι πολλά, υπάρχει μοχλός· αν είναι λίγα,
δεν υπάρχει.

ΠΡΟΣΟΧΗ ΣΤΟ QUOTA: κάθε ερώτηση = 1 κλήση Gemini (free tier ~20/ημέρα).
Default 3 ερωτήσεις. ΜΗΝ το τρέχεις αλόγιστα.

    docker compose exec backend python evaluation/measure_e2e.py
    docker compose exec backend python evaluation/measure_e2e.py --n 5
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_set_50.jsonl")


def load_questions(n: int) -> list:
    out = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("category") != "out_of_corpus":
                out.append(t["question"])
    return out[:n]


async def one(question: str) -> dict:
    """Μία πλήρης ερώτηση μέσα από το ask_ai — ό,τι ακριβώς τρέχει το /chat."""
    t_start = time.perf_counter()
    first_token = None
    chars = 0
    metrics = {}
    async for packet in ai_core.ask_ai(question=question, target_filenames=None,
                                       history=None, user_id=None):
        if packet["type"] == "text":
            if first_token is None:
                first_token = time.perf_counter() - t_start
            chars += len(packet["data"])
        elif packet["type"] == "metrics":
            metrics = packet["data"]
    total = time.perf_counter() - t_start
    return {"total": total, "ttft": first_token or total, "chars": chars,
            **metrics}


async def main_async(n: int) -> int:
    questions = load_questions(n)
    print(f"Ερωτήσεις: {len(questions)} · μοντέλο {ai_core.GEMINI_MODEL}")
    print("ΠΡΟΣΟΧΗ: καίει quota Gemini (1 κλήση ανά ερώτηση).\n")

    rows = []
    for i, q in enumerate(questions, 1):
        r = await one(q)
        rows.append(r)
        # Thinking tokens = ό,τι χρεώθηκε στην έξοδο αλλά ΔΕΝ βγήκε ως κείμενο.
        # Χονδρική εκτίμηση από τους χαρακτήρες (~4 χαρ./token στα αγγλικά).
        visible = r["chars"] / 4
        out_tok = r.get("completion_tokens") or 0
        thinking = max(0, out_tok - visible)
        r["thinking_est"] = thinking
        print(f"{i}. {q[:52]:<54}")
        print(f"   συνολικά {r['total']:>5.2f}s · TTFT {r['ttft']:>5.2f}s · "
              f"retrieval {r.get('retrieval_s', 0):>5.2f}s · "
              f"generation {r.get('generation_s', 0):>5.2f}s")
        print(f"   tokens: prompt {r.get('prompt_tokens')} · "
              f"completion {out_tok} · ορατά ~{visible:.0f} · "
              f"thinking ~{thinking:.0f}")

    n_ok = len(rows)
    avg = lambda k: sum(r.get(k) or 0 for r in rows) / n_ok  # noqa: E731
    print("\n--- ΜΕΣΟΙ ΟΡΟΙ ---")
    tot, ret, gen = avg("total"), avg("retrieval_s"), avg("generation_s")
    print(f"  συνολικά      {tot:>6.2f}s")
    print(f"  retrieval     {ret:>6.2f}s  ({100 * ret / tot:>4.1f}%)")
    print(f"  generation    {gen:>6.2f}s  ({100 * gen / tot:>4.1f}%)")
    print(f"  TTFT          {avg('ttft'):>6.2f}s")
    print(f"  prompt tokens {avg('prompt_tokens'):>6.0f}")
    print(f"  thinking est. {avg('thinking_est'):>6.0f} tokens")

    print("\nΣΥΜΠΕΡΑΣΜΑ:")
    if ret / tot < 0.15:
        print(f"  Το retrieval είναι μόλις {100 * ret / tot:.0f}% του χρόνου. "
              f"Περαιτέρω βελτιστοποίησή του ΔΕΝ γίνεται αντιληπτή από τον "
              f"χρήστη — ο μοχλός είναι η γέννηση.")
    else:
        print(f"  Το retrieval είναι ακόμα {100 * ret / tot:.0f}% — αξίζει "
              f"περαιτέρω δουλειά εκεί.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="πλήθος ερωτήσεων (quota!)")
    return asyncio.run(main_async(ap.parse_args().n))


if __name__ == "__main__":
    sys.exit(main())
