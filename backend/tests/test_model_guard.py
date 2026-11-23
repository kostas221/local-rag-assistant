"""Το model_guard σταματά μετρήσεις με όνομα μοντέλου που από 1/12/2026 ανακατευθύνεται σιωπηλά."""
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))

import model_guard


def test_before_redirect_date_is_allowed():
    model_guard.ensure_not_redirected("gemini-2.5-flash", "κριτής", today=date(2026, 11, 30))


def test_on_redirect_date_raises_and_names_the_target(monkeypatch):
    monkeypatch.delenv("ALLOW_REDIRECTED_MODEL", raising=False)
    with pytest.raises(RuntimeError, match=r"gemini-3\.5-flash-lite"):
        model_guard.ensure_not_redirected("gemini-2.5-flash", "κριτής", today=date(2026, 12, 1))


def test_explicit_model_and_override_are_allowed(monkeypatch):
    model_guard.ensure_not_redirected("gemini-3.8-flash", "σύστημα", today=date(2027, 1, 1))
    monkeypatch.setenv("ALLOW_REDIRECTED_MODEL", "1")
    model_guard.ensure_not_redirected("gemini-2.5-flash", "κριτής", today=date(2027, 1, 1))
