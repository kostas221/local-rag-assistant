"""Unit tests για το REST μονοπάτι γέννησης (gemini_rest.py).

ΓΙΑΤΙ ΕΧΟΥΝ ΑΞΙΑ: το gemini_rest αντικατέστησε το SDK στο ΚΡΙΣΙΜΟ μονοπάτι —
ό,τι σπάσει εδώ, σπάει κάθε απάντηση. Και δεν χρειάζεται ούτε API key ούτε
μοντέλα: το httpx client αντικαθίσταται με ψεύτικο που γυρνά προκαθορισμένες
γραμμές SSE. Άρα ΔΕΝ πληρώνουν το 40-60s import του ai_core ούτε καίνε quota,
και μπαίνουν στο γρήγορο job του CI.

ΤΙ ΠΡΟΣΤΑΤΕΥΟΥΝ ΣΥΓΚΕΚΡΙΜΕΝΑ: ότι τα thinking parts ΔΕΝ φτάνουν στον χρήστη
(είναι η εσωτερική σκέψη του μοντέλου, όχι απάντηση) και ότι ο υπολογισμός των
κρυφών thinking tokens από τη διαφορά total-prompt-candidates μένει σωστός —
αυτή η αριθμητική ήταν που αποκάλυψε τα 889 tokens που δεν φαίνονταν πουθενά.
"""
import asyncio
import json

import pytest

import gemini_rest

# --- Ψεύτικο httpx: γυρνά προκαθορισμένες γραμμές SSE ------------------------

class _FakeResponse:
    def __init__(self, lines, status=200):
        self.status_code = status
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b'{"error": {"message": "boom"}}'


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, lines, status=200, captured=None):
        self._lines, self._status, self._captured = lines, status, captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, _method, url, json=None, headers=None):
        if self._captured is not None:
            self._captured["url"] = url
            self._captured["body"] = json
            self._captured["headers"] = headers
        return _FakeStream(_FakeResponse(self._lines, self._status))


def _patch_httpx(monkeypatch, lines, status=200, captured=None):
    monkeypatch.setattr(
        gemini_rest.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient(lines, status, captured))


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def _text_chunk(text: str, thought: bool = False) -> str:
    part = {"text": text}
    if thought:
        part["thought"] = True
    return _sse({"candidates": [{"content": {"parts": [part]}}]})


def _collect(**kwargs):
    """Τρέχει τον async generator και μαζεύει (κείμενο, usage)."""
    async def run():
        texts, usage = [], None
        async for kind, data in gemini_rest.stream_generate(
                "prompt", model="m", api_key="k", **kwargs):
            if kind == "text":
                texts.append(data)
            else:
                usage = data
        return "".join(texts), usage
    return asyncio.run(run())


# --- Streaming & φιλτράρισμα thinking ---------------------------------------

def test_streams_text_parts_in_order(monkeypatch):
    _patch_httpx(monkeypatch, [_text_chunk("Hello "), _text_chunk("world")])
    text, _ = _collect()
    assert text == "Hello world"


def test_thought_parts_are_never_yielded(monkeypatch):
    """Η ΚΑΡΔΙΑ της αλλαγής: τα thinking parts είναι εσωτερική σκέψη του
    μοντέλου. Αν διαρρεύσουν στο UI, ο χρήστης βλέπει τον συλλογισμό αντί για
    την απάντηση."""
    _patch_httpx(monkeypatch, [
        _text_chunk("σκέψη που δεν πρέπει να φανεί", thought=True),
        _text_chunk("Η απάντηση."),
    ])
    text, _ = _collect()
    assert text == "Η απάντηση."


def test_malformed_sse_line_is_skipped(monkeypatch):
    """Keep-alive ή μισογραμμένη γραμμή δεν πρέπει να ρίχνει ΟΛΗ την απάντηση."""
    _patch_httpx(monkeypatch, [
        "data: {όχι έγκυρο json",
        ": keep-alive σχόλιο",
        "",
        _text_chunk("εντάξει"),
    ])
    text, _ = _collect()
    assert text == "εντάξει"


def test_usage_metadata_is_emitted(monkeypatch):
    _patch_httpx(monkeypatch, [
        _text_chunk("απάντηση"),
        _sse({"usageMetadata": {"promptTokenCount": 100,
                                "candidatesTokenCount": 20,
                                "totalTokenCount": 200}}),
    ])
    text, usage = _collect()
    assert text == "απάντηση"
    assert usage["promptTokenCount"] == 100


# --- Το body του αιτήματος ---------------------------------------------------

def test_thinking_budget_is_sent_when_given(monkeypatch):
    captured = {}
    _patch_httpx(monkeypatch, [_text_chunk("x")], captured=captured)
    _collect(thinking_budget=512)
    cfg = captured["body"]["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingBudget": 512}


def test_thinking_budget_zero_is_sent_not_dropped(monkeypatch):
    """REGRESSION: το 0 είναι falsy. Ένα `if thinking_budget:` θα το έριχνε
    σιωπηλά και το μοντέλο θα ξανάσκεφτόταν ελεύθερα — με τον χρήστη να νομίζει
    ότι το έσβησε."""
    captured = {}
    _patch_httpx(monkeypatch, [_text_chunk("x")], captured=captured)
    _collect(thinking_budget=0)
    assert (captured["body"]["generationConfig"]["thinkingConfig"]
            == {"thinkingBudget": 0})


def test_no_thinking_config_when_none(monkeypatch):
    """None = άσε το μοντέλο να αποφασίσει -> το πεδίο ΔΕΝ μπαίνει καθόλου."""
    captured = {}
    _patch_httpx(monkeypatch, [_text_chunk("x")], captured=captured)
    _collect(thinking_budget=None)
    assert "thinkingConfig" not in captured["body"]["generationConfig"]


def test_api_key_goes_in_header_not_url(monkeypatch):
    """ΑΣΦΑΛΕΙΑ: κλειδί σε query string καταλήγει σε logs proxy και server."""
    captured = {}
    _patch_httpx(monkeypatch, [_text_chunk("x")], captured=captured)
    _collect()
    assert captured["headers"]["x-goog-api-key"] == "k"
    assert "k" not in captured["url"].split("?")[-1]


# --- Σφάλματα ---------------------------------------------------------------

def test_non_429_http_error_raises_immediately(monkeypatch):
    """400/403 δεν διορθώνονται με retry — μην καθυστερείς τον χρήστη 62s."""
    _patch_httpx(monkeypatch, [], status=400)
    with pytest.raises(RuntimeError, match="400"):
        _collect()


# --- usage_tokens ------------------------------------------------------------

def test_usage_tokens_derives_thinking_from_difference():
    """Τα thinking tokens συχνά ΔΕΝ επιστρέφονται ρητά — βγαίνουν από τη
    διαφορά. Ακριβώς έτσι βρέθηκαν τα 889 κρυφά tokens."""
    prompt, completion, thinking = gemini_rest.usage_tokens(
        {"promptTokenCount": 10011, "candidatesTokenCount": 134,
         "totalTokenCount": 11034})
    assert (prompt, completion, thinking) == (10011, 134, 889)


def test_usage_tokens_prefers_explicit_field():
    _, _, thinking = gemini_rest.usage_tokens(
        {"promptTokenCount": 100, "candidatesTokenCount": 10,
         "totalTokenCount": 500, "thoughtsTokenCount": 390})
    assert thinking == 390


def test_usage_tokens_never_negative():
    """Αν το total λείπει, η αφαίρεση θα έβγαζε αρνητικό — και θα εμφανιζόταν
    ως παράλογο νούμερο στα metrics του UI."""
    _, _, thinking = gemini_rest.usage_tokens(
        {"promptTokenCount": 100, "candidatesTokenCount": 10})
    assert thinking == 0
