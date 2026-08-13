"""Streaming γέννηση από το Gemini μέσω REST (v1beta SSE), χωρίς SDK.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ — τρία προβλήματα, μία λύση:

1. THINKING TOKENS. Το gemini-2.5-flash είναι thinking model: σε factual ερώτηση
   με ~10.000 tokens context παρήγαγε ~900-1.700 thinking tokens ΠΡΙΝ τον πρώτο
   ορατό χαρακτήρα. Μετρημένο (evaluation/trace_stream.py): TTFT 4.81s από 5.12s
   συνολικά -> ο χρήστης κοίταζε κενή οθόνη για το 94% της αναμονής, παρότι ο
   κώδικας έκανε streaming σωστά. Τα tokens αυτά χρεώνονται κανονικά.

2. ΤΟ SDK ΔΕΝ ΜΠΟΡΕΙ ΝΑ ΤΟ ΡΥΘΜΙΣΕΙ. Το google.generativeai 0.8.6 — και το
   υποκείμενο protobuf GenerationConfig — ΔΕΝ έχει πεδίο thinking_config·
   είναι προγενέστερο των thinking models. Επιβεβαιωμένο με introspection.

3. ΤΟ MIGRATION ΕΙΝΑΙ ΜΠΛΟΚΑΡΙΣΜΕΝΟ. Το google-genai απαιτεί pydantic>=2.12.5,
   ενώ το project είναι καρφωμένο σε Pydantic V1 (FastAPI 0.99.1).

Το REST endpoint δέχεται thinkingConfig κατευθείαν και χρειάζεται ΜΟΝΟ httpx,
που είναι ήδη εγκατεστημένο. Δηλαδή βγαίνουμε από το deprecated SDK για το
κρίσιμο μονοπάτι ΧΩΡΙΣ να αγγίξουμε το pydantic pin.

ΜΕΤΡΗΜΕΝΟ (evaluation/compare_thinking_budgets.py, ίδιο prompt & context):
  enumeration (n=4)        control TTFT 5.03s -> budget=0 1.20s  (4.19x)
  reasoning/multi_hop (n=3) control TTFT 7.66s -> budget=0 1.14s  (6.71x)
  keyword coverage της ΑΠΑΝΤΗΣΗΣ: ΤΑΥΤΟΣΗΜΟ (100% / 66.7%) και στις δύο ομάδες.

ΜΠΟΝΟΥΣ ΑΚΡΙΒΕΙΑΣ: το REST σημαδεύει τα thinking parts με `"thought": true`,
οπότε τα ξεχωρίζουμε ΡΗΤΑ αντί για το heuristic του _safe_chunk_text (που
έπιανε το thinking μόνο έμμεσα, μέσω exception στο chunk.text).
"""
import asyncio
import json

import httpx
from loguru import logger

ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:streamGenerateContent?alt=sse")

# Ξεχωριστά timeouts: το connect πρέπει να αποτύχει γρήγορα, αλλά το read
# περιμένει ολόκληρη τη γέννηση — με μεγάλο context και πολλά output tokens
# ένα ενιαίο timeout είτε κόβει σωστές απαντήσεις είτε κρεμάει σε νεκρό socket.
TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)


async def stream_generate(prompt: str, *, model: str, api_key: str,
                          temperature: float = 0.1,
                          max_output_tokens: int = 4096,
                          thinking_budget: int | None = 0,
                          retries: int = 5):
    """Async generator που δίνει ('text', str) και στο τέλος ('usage', dict).

    thinking_budget: 0 = σβηστό (default), None = άσε το μοντέλο να αποφασίσει
    (η προηγούμενη συμπεριφορά), θετικός αριθμός = ανώτατο όριο thinking tokens.

        Retry σε 429/5xx/δικτυακά σφάλματα με τα ΙΔΙΑ waits (2,4,8,16,32s) που είχε
    το _gemini_generate — καλύπτουν ένα ολόκληρο per-minute rate-limit window.
    Αν ο server στείλει Retry-After, υπερισχύει ΜΟΝΟ όταν ζητάει περισσότερο.
    ΠΡΟΣΟΧΗ: το retry γίνεται ΜΟΝΟ πριν σταλεί το πρώτο κομμάτι κειμένου. Αν η
    σύνδεση σπάσει στη μέση, δεν ξαναρχίζουμε — ο χρήστης έχει ήδη δει μισή
    απάντηση και μια δεύτερη θα την κολλούσε από πάνω.
    """
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if thinking_budget is not None:
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget}

    url = ENDPOINT.format(model=model)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    for attempt in range(retries):
        emitted = False
        try:
            async with (
                httpx.AsyncClient(timeout=TIMEOUT) as client,
                client.stream("POST", url, json=body, headers=headers) as response,
            ):
                if response.status_code != 200:
                    raw = (await response.aread())[:300].decode(errors="replace")
                    hint = _retry_after(response.headers)
                    if response.status_code == 429:
                        raise _RateLimited(raw, hint)
                    # 5xx = ΠΑΡΟΔΙΚΟ πρόβλημα της άλλης πλευράς. Το 503
                    # UNAVAILABLE ("model overloaded") είναι το συχνότερο
                    # σφάλμα του Gemini σε ώρα αιχμής και περνάει με ένα retry
                    # των 2s. Τα 4xx είναι ΔΙΚΟ ΜΑΣ λάθος (400 κακό body, 403
                    # άκυρο κλειδί) -> retry = 62s καθυστέρηση πριν το ίδιο
                    # ακριβώς σφάλμα.
                    if response.status_code >= 500:
                        raise _ServerError(
                            f"Gemini REST {response.status_code}: {raw}", hint)
                    raise RuntimeError(
                        f"Gemini REST {response.status_code}: {raw}")

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        payload = json.loads(line[6:])
                    except ValueError:
                        continue  # keep-alive / μισή γραμμή -> αγνόησέ τη
                    for cand in payload.get("candidates", []):
                        content = cand.get("content") or {}
                        for part in content.get("parts") or []:
                            # thought=True -> εσωτερική σκέψη, ΟΧΙ απάντηση
                            if part.get("thought"):
                                continue
                            text = part.get("text")
                            if text:
                                emitted = True
                                yield ("text", text)
                    if "usageMetadata" in payload:
                        yield ("usage", payload["usageMetadata"])
            return
        except asyncio.CancelledError:
            # Ο χρήστης έκλεισε τη σύνδεση -> προς τα πάνω, ώστε το FastAPI να
            # κλείσει το socket και να μη χρεωνόμαστε άλλα tokens (FinOps).
            raise
        except (_Retryable, httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.RemoteProtocolError, httpx.ConnectError) as e:
            if emitted or attempt == retries - 1:
                raise
            wait = min(32, 2 ** (attempt + 1))  # 2,4,8,16,32
            # Το Retry-After υπερισχύει ΜΟΝΟ προς τα πάνω: ένα μικρό header θα
            # μας έβαζε να ξαναχτυπήσουμε νωρίτερα απ' ό,τι αντέχει ο ίδιος ο
            # server. Καπάκι 60s ώστε ένα παράλογο header να μη κρεμάσει το
            # αίτημα πίσω από ένα ήδη ανοιχτό HTTP connection του χρήστη.
            hint = getattr(e, "retry_after", None)
            if hint is not None:
                wait = min(60.0, max(wait, hint))
            logger.warning(f"Gemini REST ({type(e).__name__}). Retry σε {wait}s... "
                           f"[{attempt + 1}/{retries}]")
            await asyncio.sleep(wait)
    raise RuntimeError("Gemini REST: εξάντληση retries.")


class _Retryable(Exception):
    """Παροδικό σφάλμα του API — μπαίνει στο backoff του stream_generate.

    Κουβαλάει το Retry-After του server (δευτερόλεπτα ή None) ώστε η αναμονή
    να είναι πληροφορημένη αντί για εικασία.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class _RateLimited(_Retryable):
    """429 — ξεπεράσαμε το quota ΜΑΣ."""


class _ServerError(_Retryable):
    """5xx — προσωρινό πρόβλημα στη μεριά του Google (503 = overloaded)."""


def _retry_after(headers) -> float | None:
    """Το Retry-After σε δευτερόλεπτα, αν ήρθε και είναι αριθμός.

    Η μορφή HTTP-date ΔΕΝ καλύπτεται σκόπιμα: το Gemini στέλνει δευτερόλεπτα,
    και ένα λάθος parse ημερομηνίας θα έδινε λάθος αναμονή αντί για καμία.
    """
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


async def generate_once(prompt: str, *, model: str, api_key: str,
                        temperature: float = 0.1,
                        max_output_tokens: int = 256,
                        thinking_budget: int | None = 0,
                        retries: int = 5) -> str:
    """Μία σύντομη, ΜΗ streaming απάντηση (μετάφραση ερωτήματος, query rewrite).

    Χτίζεται πάνω στο stream_generate αντί για ξεχωριστό endpoint: ένα μονοπάτι
    δικτύου, ένα retry/backoff, ένα σημείο όπου μπορεί να μπει bug.

    ΤΑ DEFAULTS ΕΙΝΑΙ ΣΚΟΠΙΜΑ ΣΦΙΧΤΑ. Αυτές οι κλήσεις παράγουν ~10 λέξεις:
      • thinking_budget=0 — μια μετάφραση δεν χρειάζεται συλλογισμό, και το
        query rewrite τρέχει σε ΚΑΘΕ follow-up ερώτηση (δεν είναι cached όπως οι
        μεταφράσεις). Με το SDK πλήρωνε ανεξέλεγκτα thinking tokens για να βγάλει
        μια γραμμή κειμένου.
      • max_output_tokens=256 αντί για 4096 — καπάκι κόστους χωρίς κανένα ρίσκο
        κοψίματος σε έξοδο αυτού του μεγέθους.
    """
    parts = []
    async for kind, data in stream_generate(
            prompt, model=model, api_key=api_key, temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget, retries=retries):
        if kind == "text":
            parts.append(data)
    return "".join(parts)


def usage_tokens(usage: dict) -> tuple:
    """(prompt_tokens, completion_tokens, thinking_tokens) από το usageMetadata.

    Το thoughts token count δεν επιστρέφεται πάντα ρητά· όταν λείπει, βγαίνει
    από τη διαφορά total - prompt - candidates. Αυτή ακριβώς η αριθμητική
    αποκάλυψε τα 889 κρυφά tokens που κινητοποίησαν όλη αυτή την αλλαγή."""
    prompt = usage.get("promptTokenCount") or 0
    completion = usage.get("candidatesTokenCount") or 0
    total = usage.get("totalTokenCount") or 0
    thinking = usage.get("thoughtsTokenCount")
    if thinking is None:
        thinking = max(0, total - prompt - completion)
    return prompt, completion, thinking
