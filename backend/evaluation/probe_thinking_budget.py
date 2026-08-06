"""Μπορούμε να σβήσουμε το «thinking» του 2.5-flash, και τι κοστίζει;

ΤΟ ΠΡΟΒΛΗΜΑ (μετρημένο, trace_stream.py): σε factual ερώτηση με 10.011 tokens
context, το μοντέλο παράγει ~889 thinking tokens ΠΡΙΝ τον πρώτο ορατό
χαρακτήρα. Αποτέλεσμα: TTFT 4.81s από 5.12s συνολικά -> ο χρήστης κοιτάει κενή
οθόνη για το 94% της αναμονής, παρότι ο κώδικας κάνει streaming σωστά.

ΓΙΑΤΙ ΔΕΝ ΛΥΝΕΤΑΙ ΜΕ ΤΟ SDK: το google.generativeai 0.8.6 (και το υποκείμενο
protobuf) ΔΕΝ έχει πεδίο thinking_config — είναι προγενέστερο των thinking
models. Το migration στο google-genai μπλοκάρεται από το pydantic V1 pin.
ΤΡΙΤΟΣ ΔΡΟΜΟΣ: το REST endpoint v1beta δέχεται thinkingConfig κατευθείαν, χωρίς
SDK. Αυτό το script ελέγχει αν όντως δουλεύει ΚΑΙ τι κάνει στην απάντηση.

ΤΡΕΙΣ ΣΥΝΘΗΚΕΣ (ίδιο prompt, ίδιο context, ίδιο temperature):
  A. REST χωρίς thinkingConfig  -> control: το REST συμπεριφέρεται σαν το SDK;
  B. REST με thinkingBudget=0   -> thinking σβηστό
  C. REST με thinkingBudget=512 -> ενδιάμεσο

ΚΡΙΝΕΤΑΙ: TTFT, συνολικός χρόνος, thinking tokens (από τη διαφορά του total)
ΚΑΙ το ίδιο το κείμενο της απάντησης — γιατί η ταχύτητα χωρίς την απάντηση δεν
σημαίνει τίποτα. Η ποιότητα κρίνεται σοβαρά μόνο με judge run· εδώ κοιτάμε αν
υπάρχει προφανής υποβάθμιση (κομμένη/κενή/λάθος γλώσσα).

ΚΟΣΤΟΣ: 3 κλήσεις Gemini.

    docker compose exec backend python evaluation/probe_thinking_budget.py
"""
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

QUESTION = ("What is the data transfer bottleneck obstacle and a proposed "
            "solution?")
ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:streamGenerateContent?alt=sse")


async def build_prompt() -> str:
    pages = await ai_core.search_documents(QUESTION)
    parts = []
    for text, meta in pages:
        parts.append("\n[Source: {}, Page: {}]\n{}\n".format(
            meta.get("file_name"), meta.get("page"), text))
    return ("Answer using ONLY the source text below.\n"
            "--- SOURCE TEXT ---\n" + "".join(parts) +
            "\n\nQUESTION: " + QUESTION + "\n\nANSWER:")


async def call_rest(prompt: str, thinking_budget=None, retries: int = 3) -> dict:
    """Μία streaming κλήση μέσω REST (SSE). Επιστρέφει χρόνους + tokens + κείμενο.

    RETRY: το endpoint κρεμάει περιστασιακά (ReadTimeout) και σε 429 κάτω από
    rate limit. Χωρίς retry ένα αργό request σκότωνε ολόκληρο πείραμα ΑΦΟΥ είχε
    ήδη ξοδέψει quota στις προηγούμενες ερωτήσεις. Το backoff (4,8,16s) καλύπτει
    ένα per-minute παράθυρο, ίδια λογική με το _gemini_generate του ai_core."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }
    if thinking_budget is not None:
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget}

    url = ENDPOINT.format(model=ai_core.GEMINI_MODEL)
    headers = {"x-goog-api-key": ai_core.GEMINI_API_KEY,
               "Content-Type": "application/json"}

    # Ξεχωριστά timeouts: το connect πρέπει να είναι σύντομο, αλλά το read
    # περιμένει ΟΛΟΚΛΗΡΗ τη σκέψη του μοντέλου πριν έρθει το πρώτο byte.
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)
    last_error = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        ttft, chunks, text_out, usage = None, 0, [], {}
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, json=body, headers=headers) as r,
            ):
                if r.status_code != 200:
                    raw = await r.aread()
                    last_error = f"HTTP {r.status_code}: {raw[:200].decode()}"
                    if r.status_code != 429:
                        return {"error": last_error}
                    raise httpx.ReadTimeout(last_error)  # -> backoff
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    chunks += 1
                    for cand in payload.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            # thought=True -> thinking part, ΟΧΙ ορατό κείμενο
                            if part.get("thought"):
                                continue
                            t = part.get("text")
                            if t:
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                text_out.append(t)
                    if "usageMetadata" in payload:
                        usage = payload["usageMetadata"]
            break
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt == retries - 1:
                return {"error": last_error}
            wait = 2 ** (attempt + 2)  # 4, 8, 16
            print(f"     (retry σε {wait}s — {type(e).__name__})")
            await asyncio.sleep(wait)
    total = time.perf_counter() - t0
    prompt_t = usage.get("promptTokenCount", 0)
    cand_t = usage.get("candidatesTokenCount", 0)
    total_t = usage.get("totalTokenCount", 0)
    return {
        "ttft": ttft, "total": total, "chunks": chunks,
        "text": "".join(text_out),
        "prompt_tokens": prompt_t, "completion_tokens": cand_t,
        # Το REST εκθέτει thoughtsTokenCount· αν λείπει, το βγάζουμε από τη διαφορά
        "thinking": usage.get("thoughtsTokenCount",
                              max(0, total_t - prompt_t - cand_t)),
    }


async def main_async() -> int:
    prompt = await build_prompt()
    print(f"prompt: {len(prompt)} χαρακτήρες · μοντέλο {ai_core.GEMINI_MODEL}\n")

    conditions = [
        ("A. χωρίς thinkingConfig", None),
        ("B. thinkingBudget=0", 0),
        ("C. thinkingBudget=512", 512),
    ]
    results = {}
    for label, budget in conditions:
        r = await call_rest(prompt, budget)
        results[label] = r
        if "error" in r:
            print(f"{label:<26} ΣΦΑΛΜΑ: {r['error']}\n")
            continue
        print(f"{label:<26} TTFT {r['ttft'] or -1:>5.2f}s · "
              f"σύνολο {r['total']:>5.2f}s · chunks {r['chunks']:>3} · "
              f"thinking {r['thinking']:>4} tok · "
              f"απάντηση {r['completion_tokens']:>4} tok")

    ok = {k: v for k, v in results.items() if "error" not in v and v.get("ttft")}
    if len(ok) >= 2:
        base = results.get("A. χωρίς thinkingConfig", {})
        print("\n--- Επιτάχυνση έναντι του control (A) ---")
        for label, r in ok.items():
            if base.get("ttft"):
                print(f"  {label:<26} TTFT {base['ttft'] / r['ttft']:>5.2f}x · "
                      f"σύνολο {base['total'] / r['total']:>5.2f}x")

    print("\n--- ΑΠΑΝΤΗΣΕΙΣ (πρώτοι 320 χαρακτήρες) ---")
    for label, r in results.items():
        if "error" in r:
            continue
        print(f"\n[{label}] {len(r['text'])} χαρακτήρες")
        print("  " + r["text"][:320].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
