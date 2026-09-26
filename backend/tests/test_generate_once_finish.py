"""Το generate_once ΔΕΝ επιστρέφει μισή απάντηση ως ολόκληρη.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (25/9/2026): η μετάφραση της q025 («Ποιες τιμές ηλεκτρικού ρεύματος
ανά κιλοβατώρα αναφέρει το paper…») βγήκε «cost per» και ΚΛΕΙΔΩΘΗΚΕ στο translation
cache — κάθε επόμενο τρέξιμο έψαχνε με δύο λέξεις (MRR 1.0 -> 0.33). Το ίδιο prompt
5 φορές έδωσε 5/5 ολόκληρες με finishReason=STOP: σπάνια κακή κλήση, που περνούσε
σιωπηλά επειδή κανείς δεν κοίταζε ΓΙΑΤΙ σταμάτησε το μοντέλο.

ΤΙ ΠΡΟΣΤΑΤΕΥΟΥΝ:
  • ολόκληρη απάντηση (STOP) περνάει αυτούσια
  • ροή που κόβεται χωρίς finishReason -> σφάλμα, όχι μισό κείμενο
  • finishReason άλλο από STOP (MAX_TOKENS, SAFETY…) -> σφάλμα
  • το stream_generate ΔΕΝ δίνει ('finish', …) χωρίς ρητό with_finish=True: ο
    καταναλωτής της γέννησης (ai_core, `else: usage_tokens(data)`) θεωρεί ό,τι δεν
    είναι 'text' ως usage και θα έσκαγε.

Χωρίς API key, χωρίς μοντέλα — ψεύτικο httpx, όπως στο test_gemini_rest.py.
"""
import asyncio
import json

import pytest

import gemini_rest


class _FakeResponse:
    def __init__(self, lines):
        self.status_code = 200
        self.headers = {}
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, _method, _url, json=None, headers=None):
        return _FakeStream(_FakeResponse(self._lines))


def _patch(monkeypatch, lines):
    monkeypatch.setattr(gemini_rest.httpx, "AsyncClient", lambda **kw: _FakeClient(lines))


def _chunk(text: str, finish: str | None = None) -> str:
    cand = {"content": {"parts": [{"text": text}]}}
    if finish:
        cand["finishReason"] = finish
    return "data: " + json.dumps({"candidates": [cand]})


def _once():
    return asyncio.run(gemini_rest.generate_once("prompt", model="m", api_key="k"))


def test_complete_answer_is_returned(monkeypatch):
    _patch(monkeypatch, [_chunk("What cost per kilowatt-hour "),
                         _chunk("does the paper report?", finish="STOP")])
    assert _once() == "What cost per kilowatt-hour does the paper report?"


def test_stream_cut_without_finish_reason_raises(monkeypatch):
    """Ο μηχανισμός της q025: κείμενο ήρθε, λόγος τερματισμού ΔΕΝ ήρθε."""
    _patch(monkeypatch, [_chunk("cost per")])
    with pytest.raises(RuntimeError, match="ημιτελής"):
        _once()


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"])
def test_non_stop_finish_reason_raises(monkeypatch, reason):
    _patch(monkeypatch, [_chunk("cost per", finish=reason)])
    with pytest.raises(RuntimeError, match=reason):
        _once()


def test_stream_generate_hides_finish_by_default(monkeypatch):
    """Η γέννηση απαντήσεων δεν ζητάει with_finish -> δεν πρέπει να δει ποτέ 'finish'."""
    _patch(monkeypatch, [_chunk("Hello ", finish=None), _chunk("world", finish="STOP")])

    async def run():
        return [kind async for kind, _data in gemini_rest.stream_generate(
            "prompt", model="m", api_key="k")]
    assert asyncio.run(run()) == ["text", "text"]
