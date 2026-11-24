"""Το πάγωμα του scoreboard (evaluation/scoreboard.py, Frozen). Χωρίς μοντέλα -> γρήγορα tests.

--reuse-translations (24/11/2026, κανόνας Α3 του 5α): όταν αλλάζει ΜΟΝΟ η σειρά όρων στο γλωσσάρι, η
μετάφραση της ίδιας ερώτησης ξαναχρησιμοποιείται· οτιδήποτε άλλο λείπει μένει μπλοκαρισμένο με --frozen-only."""
import asyncio
import hashlib
import json
import os
import sys

import pytest

EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
sys.path.insert(0, EVAL)

import scoreboard as S  # noqa: E402

import gemini_rest  # noqa: E402

OLD = ("You are a translation assistant for an English-only academic search engine. The corpus is about: "
       "Cloud. Corpus glossary: map, input, machines. User question: Τι είναι το MapReduce;")
NEW = OLD.replace("input, machines", "machines, input")


def _frozen(tmp_path, monkeypatch, reuse):
    path = tmp_path / "frozen.json"
    key = hashlib.sha256(OLD.encode("utf-8")).hexdigest()
    path.write_text(json.dumps({key: {"tail": S.safe_tail(OLD), "out": "what is mapreduce"}}), encoding="utf-8")
    fr = S.Frozen(str(path), fresh=False, frozen_only=True, reuse_translations=reuse)

    async def never(prompt, **kw):
        raise AssertionError("δεν πρέπει να γίνει πραγματική κλήση")

    # monkeypatch: επαναφέρει το ΠΡΑΓΜΑΤΙΚΟ generate_once μετά το test (το install το αντικαθιστά)
    monkeypatch.setattr(gemini_rest, "generate_once", never)
    fr.install()
    return fr


def test_reordered_glossary_reuses_the_same_questions_translation(tmp_path, monkeypatch):
    fr = _frozen(tmp_path, monkeypatch, reuse=True)
    assert asyncio.run(gemini_rest.generate_once(NEW)) == "what is mapreduce"
    assert fr.reused == 1 and not fr.errors


def test_without_the_flag_the_changed_prompt_is_blocked(tmp_path, monkeypatch):
    fr = _frozen(tmp_path, monkeypatch, reuse=False)
    with pytest.raises(RuntimeError):
        asyncio.run(gemini_rest.generate_once(NEW))
    assert fr.errors


def test_other_question_or_other_prompt_is_never_reused(tmp_path, monkeypatch):
    fr = _frozen(tmp_path, monkeypatch, reuse=True)
    with pytest.raises(RuntimeError):
        asyncio.run(gemini_rest.generate_once(NEW.replace("Τι είναι το MapReduce;", "Τι είναι το PyWren;")))
    with pytest.raises(RuntimeError):
        asyncio.run(gemini_rest.generate_once("Rewrite the query. Original query: Τι είναι το MapReduce;"))
    assert fr.reused == 0


def test_long_question_stored_in_the_old_tail_format_is_reused(tmp_path, monkeypatch):
    """Παλιά μορφή (πριν από τις 23/11): tail = οι τελευταίοι 200 χαρακτήρες· σε μακριά ερώτηση χωρίς
    το «User question:». Στο 1ο τρέξιμο του Α3 αυτό άφησε 24 μακριές ελληνικές ερωτήσεις αμετάφραστες."""
    long_q = "Πώς διαχειρίζεται το MapReduce την επαναπροσδιορισμένη εργασία " * 5
    old = OLD.replace("Τι είναι το MapReduce;", long_q)
    new = old.replace("input, machines", "machines, input")
    path = tmp_path / "frozen.json"
    key = hashlib.sha256(old.encode("utf-8")).hexdigest()
    path.write_text(json.dumps({key: {"tail": old[-200:], "out": "mapreduce rescheduled task"}}),
                    encoding="utf-8")
    fr = S.Frozen(str(path), fresh=False, frozen_only=True, reuse_translations=True)

    async def never(prompt, **kw):
        raise AssertionError("δεν πρέπει να γίνει πραγματική κλήση")

    monkeypatch.setattr(gemini_rest, "generate_once", never)
    fr.install()
    assert asyncio.run(gemini_rest.generate_once(new)) == "mapreduce rescheduled task"
    assert fr.reused == 1 and not fr.errors
