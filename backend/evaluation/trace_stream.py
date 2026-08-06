"""Χρονισμός ΑΝΑ CHUNK του Gemini streaming: πότε φτάνει ο πρώτος ΟΡΑΤΟΣ χαρακτήρας;

ΓΙΑΤΙ: το measure_e2e.py έδειξε TTFT 3.90s με generation 3.91s — δηλαδή ο
χρήστης βλέπει κενή οθόνη σχεδόν μέχρι το τέλος, παρότι ο κώδικας κάνει
streaming. Υπόθεση: το gemini-2.5-flash είναι thinking model και στέλνει πρώτα
chunks που περιέχουν ΜΟΝΟ thinking parts (το _safe_chunk_text επιστρέφει σωστά
"" γι' αυτά) -> τίποτα ορατό μέχρι να τελειώσει η σκέψη.

ΑΥΤΟ ΤΟ SCRIPT ΔΕΝ ΤΟ ΥΠΟΘΕΤΕΙ: τυπώνει πότε ήρθε κάθε chunk, αν είχε κείμενο,
και τα πεδία tokens του usage_metadata (όπου φαίνονται τα thoughts tokens).

ΚΟΣΤΟΣ: 1 κλήση Gemini.

    docker compose exec backend python evaluation/trace_stream.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

QUESTION = ("What is the data transfer bottleneck obstacle and a proposed "
            "solution?")


async def main_async() -> int:
    pages = await ai_core.search_documents(QUESTION)
    parts = []
    for text, meta in pages:
        header = "\n[Source: {}, Page: {}]\n".format(
            meta.get("file_name"), meta.get("page"))
        parts.append(header + text + "\n")
    context = "".join(parts)
    prompt = ("Answer using ONLY the source text below.\n"
              "--- SOURCE TEXT ---\n" + context +
              "\n\nQUESTION: " + QUESTION + "\n\nANSWER:")
    print(f"context: {len(context)} χαρακτήρες από {len(pages)} σελίδες\n")

    t0 = time.perf_counter()
    response = await ai_core._gemini_generate(prompt, stream=True)
    usage = None
    n = 0
    first_text_at = None
    async for chunk in response:
        n += 1
        elapsed = time.perf_counter() - t0
        text = ai_core._safe_chunk_text(chunk)
        if getattr(chunk, "usage_metadata", None):
            usage = chunk.usage_metadata
        if text and first_text_at is None:
            first_text_at = elapsed
        # Τα πρώτα 8 chunks πάντα, μετά μόνο δείγμα — αρκεί για το μοτίβο.
        if n <= 8 or n % 10 == 0:
            kind = "TEXT" if text else "ΚΕΝΟ(thinking;)"
            print(f"  chunk {n:>3} @ {elapsed:>6.2f}s  {kind:<16} len={len(text)}")
    total = time.perf_counter() - t0

    print(f"\n  σύνολο chunks      : {n}")
    print(f"  1ος ΟΡΑΤΟΣ χαρακτ. : {first_text_at:.2f}s")
    print(f"  συνολικός χρόνος   : {total:.2f}s")
    if first_text_at is not None:
        print(f"  ΑΝΑΜΟΝΗ ΣΤΟ ΚΕΝΟ   : {100 * first_text_at / total:.0f}% "
              f"του χρόνου")

    print("\n--- usage_metadata (πεδία tokens) ---")
    for field in dir(usage or ()):
        if not field.startswith("_") and "token" in field.lower():
            print(f"  {field} = {getattr(usage, field)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
