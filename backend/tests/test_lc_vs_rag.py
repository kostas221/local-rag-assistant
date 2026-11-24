"""Fast, model-free tests for the long-context vs RAG experiment harness."""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BACKEND_DIR, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import judge_lc_vs_rag as J  # noqa: E402
import lc_vs_rag_spec as SPEC  # noqa: E402
import probe_lc_vs_rag as P  # noqa: E402


def _synthetic_full_answers(*, include_distract: bool = True, complete: bool = True) -> dict:
    items = SPEC.build_items(SPEC.CANONICAL_FILES, include_distract=include_distract)
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for it in items:
        docs_s = ";".join(it["docs"])
        for draw in range(1, SPEC.DRAWS[it["arm"]] + 1):
            for sysk in ("full", "rag"):
                cut = int(sysk == "rag" and it["arm"] == "safety" and it["set"] == "ooc")
                if cut:
                    role = "cut"
                elif draw > 1:
                    role = "repeat"
                elif (it["arm"], docs_s, sysk) not in seen:
                    role = "first"
                    seen.add((it["arm"], docs_s, sysk))
                else:
                    role = "next"
                rows.append({
                    "arm": it["arm"], "set": it["set"], "id": it["t"]["id"], "size": it["size"],
                    "lang": SPEC.lang_of(it["t"], it["t"]["question"]),
                    "docs": docs_s, "draw": draw, "system": sysk,
                    "cache_role": role, "cut": cut, "second_search": 0,
                    "n_pages": 0 if cut else 8, "pages": "",
                    "in_tokens": None if cut else 10000,
                    "cached_tokens": 0 if cut or role == "first" else 8000,
                    "out_tokens": None if cut else 200,
                    "usd": 0.0 if cut else (0.008 if role == "first" else 0.002),
                    "fresh": 0 if cut else 1, "finish": None if cut else "STOP",
                    "ttft": None if cut else 2.5, "total_s": None if cut else 4.0,
                    "search_s": 0.8 if sysk == "rag" else None,
                    "prep_s": 0.01 if sysk == "full" else None,
                    "e2e_ttft": (0.8 if cut else 3.3) if sysk == "rag" else 2.51,
                    "e2e_live": 1, "ret_live_calls": 0, "ret_frozen_calls": 0,
                    "ret_live_s": 0.0, "ret_usd": 0.0, "refusal_re": cut,
                    "answer": "" if cut else "Sample answer.",
                })
    return {
        "meta": {
            "complete": complete,
            "include_distract": include_distract,
            "expected_rows": len(rows),
            "actual_rows": len(rows),
        },
        "rows": rows,
        "live_timing": [
            {
                "arm": "one", "set": "main", "id": "q001", "size": 1, "lang": "en",
                "docs": "EECS-2009-28.pdf",
                "rag_e2e_ttft": 3.1, "rag_search_s": 0.9, "rag_gen_ttft": 2.2,
                "rag_cut": 0, "rag_ret_calls": 0, "rag_ret_api_s": 0.0, "rag_usd": 0.01,
                "full_e2e_ttft": 2.4, "full_prep_s": 0.01, "full_gen_ttft": 2.39, "full_usd": 0.02,
            }
        ],
    }


def test_expected_row_counts_and_distractor_balance():
    primary = SPEC.expected_row_keys(include_distract=False)
    all_keys = SPEC.expected_row_keys(include_distract=True)
    assert len(primary) == 654
    assert len(all_keys - primary) == 56

    items = SPEC.build_items(SPEC.CANONICAL_FILES, include_distract=True)
    dist = [it for it in items if it["arm"] == "distract"]
    assert len(dist) == 28
    assert {it["size"] for it in dist} == {2, 3}
    # 2 unique single-paper questions per paper across all 7 papers
    by_target = {}
    for it in dist:
        by_target.setdefault(it["target_doc"], set()).add(it["t"]["id"])
    assert len(by_target) == 7
    assert all(len(qids) == 2 for qids in by_target.values())

    live = SPEC.select_live_timing_items(items, per_size=4)
    assert len(live) == 12
    for size in (1, 2, 3):
        langs = {SPEC.lang_of(it["t"], it["t"]["question"]) for it in live if it["size"] == size}
        assert langs == {"el", "en"}


def test_validate_answers_rejects_partial_and_accepts_full(tmp_path, monkeypatch):
    full = _synthetic_full_answers(include_distract=True, complete=True)
    assert SPEC.validate_answers(full, smoke=False) == []

    # 4 synthetic rows MUST fail completeness check before any OpenAI call
    tiny = {"meta": {"complete": True, "include_distract": False}, "rows": full["rows"][:4]}
    errs = SPEC.validate_answers(tiny, smoke=False)
    assert errs and any("654" in e for e in errs)

    tiny_path = tmp_path / "answers_tiny.json"
    tiny_path.write_text(json.dumps(tiny), encoding="utf-8")
    monkeypatch.setattr(J.C, "api_key", lambda: (_ for _ in ()).throw(AssertionError("should not check API")))
    assert J.main(["--input", str(tiny_path)]) == 1

    # Missing 1 row -> rejected
    one_missing = {**full, "rows": full["rows"][:-1]}
    assert SPEC.validate_answers(one_missing, smoke=False)

    # meta.complete=False -> rejected even with all rows present
    not_complete = {**full, "meta": {**full["meta"], "complete": False}}
    assert SPEC.validate_answers(not_complete, smoke=False)

    # Duplicate row -> rejected
    dup = {**full, "rows": full["rows"] + [full["rows"][0]]}
    assert SPEC.validate_answers(dup, smoke=False)

    # Empty non-cut answer -> rejected
    bad_rows = [dict(r) for r in full["rows"]]
    non_cut_idx = next(i for i, r in enumerate(bad_rows) if not r["cut"])
    bad_rows[non_cut_idx]["answer"] = "   "
    assert SPEC.validate_answers({**full, "rows": bad_rows}, smoke=False)

    # Smoke mode accepts small clean subset
    assert SPEC.validate_answers(tiny, smoke=True) == []


def test_budget_ledger_persists_across_resume_and_enforces_headroom(tmp_path):
    bpath = str(tmp_path / "budget.json")
    led = SPEC.BudgetLedger(bpath, max_usd=1.0, in_usd=0.75, out_usd=3.75, cached_share=0.10)
    led.record("gen", {"promptTokenCount": 500_000, "candidatesTokenCount": 50_000, "thoughtsTokenCount": 0})
    led.record("ret", {"promptTokenCount": 100_000, "candidatesTokenCount": 10_000, "thoughtsTokenCount": 0})
    spent_first = led.total_usd
    assert spent_first > 0.65

    # Resume: a new BudgetLedger instance MUST remember prior cumulative spend
    resumed = SPEC.BudgetLedger(bpath, max_usd=1.0, in_usd=0.75, out_usd=3.75, cached_share=0.10)
    assert resumed.total_usd == pytest.approx(spent_first)
    assert resumed.run_usd == 0.0

    # A call whose conservative worst-case estimate pushes total over $1.00 is blocked BEFORE network
    huge_prompt = "x" * 1_500_000  # ~500k tokens -> ~$0.375 > remaining ~$0.32
    with pytest.raises(RuntimeError, match="ΟΡΙΟ ΚΟΣΤΟΥΣ"):
        resumed.check_before_call(huge_prompt, 4096, label="γέννηση")
    assert resumed.exceeded is not None


def test_drawstream_isolates_retrieval_and_bills_both(tmp_path):
    bpath = str(tmp_path / "budget.json")
    gpath = str(tmp_path / "gen.json")
    budget = SPEC.BudgetLedger(bpath, max_usd=1.0, in_usd=0.75, out_usd=3.75, cached_share=0.10)
    calls = []

    async def fake_stream(prompt, **kw):
        calls.append(prompt)
        yield "text", "ok"
        if kw.get("with_finish"):
            yield "finish", "STOP"
        yield "usage", {"promptTokenCount": 10_000, "candidatesTokenCount": 1_000, "thoughtsTokenCount": 0}

    gen = P.DrawStream(fake_stream, budget, gen_path=gpath)

    async def exercise():
        # 1) Retrieval call (active=False): MUST NOT write generation_calls or bump gen.new,
        #    MUST bill budget as 'ret'
        gen.active = False
        events = [ev async for ev in gen("translate me", with_finish=True, max_output_tokens=1024)]
        assert any(k == "text" and v == "ok" for k, v in events)
        assert gen.new == 0
        assert gen.hits == 0
        assert gen.data == {}
        assert budget.state["ret_calls"] == 1
        assert budget.state["gen_calls"] == 0
        assert budget.state["ret_usd"] > 0

        # 2) Generation call (active=True): writes generation_calls and bills 'gen'
        gen.active = True
        gen.draw = 1
        assert [ev async for ev in gen("CURRENT QUESTION: hi", with_finish=True, max_output_tokens=4096)]
        assert gen.new == 1
        assert len(gen.data) == 1
        assert budget.state["gen_calls"] == 1
        total_after_first = budget.total_usd

        # 3) Replay of same generation call -> cache hit ($0), works even if max_usd is already reached
        budget.max_usd = total_after_first
        assert [ev async for ev in gen("CURRENT QUESTION: hi", with_finish=True, max_output_tokens=4096)]
        assert gen.hits == 1
        assert budget.total_usd == total_after_first
        assert len(calls) == 2

        # 4) New paid retrieval call when budget is exhausted -> blocked BEFORE calling fake_stream
        gen.active = False
        with pytest.raises(RuntimeError, match="ΟΡΙΟ ΚΟΣΤΟΥΣ"):
            _ = [ev async for ev in gen("another translation", with_finish=True, max_output_tokens=1024)]
        assert len(calls) == 2  # network was NOT called

    asyncio.run(exercise())


def test_budget_bootstrap_from_existing_caches(tmp_path):
    gpath = tmp_path / "gen.json"
    rpath = tmp_path / "ret.json"
    bpath = tmp_path / "budget.json"
    gpath.write_text(json.dumps({
        "k1": {"usage": {"promptTokenCount": 200_000, "candidatesTokenCount": 10_000, "thoughtsTokenCount": 0}}
    }), encoding="utf-8")
    rpath.write_text(json.dumps({
        "k2": {"usage": {"promptTokenCount": 50_000, "candidatesTokenCount": 2_000, "thoughtsTokenCount": 0}}
    }), encoding="utf-8")
    led = SPEC.BudgetLedger(
        str(bpath), max_usd=8.0, in_usd=0.75, out_usd=3.75, cached_share=0.10,
        seed_files=(("gen", str(gpath)), ("ret", str(rpath))),
    )
    assert led.state["gen_calls"] == 1
    assert led.state["ret_calls"] == 1
    assert led.total_usd == pytest.approx(led.state["gen_usd"] + led.state["ret_usd"])
    assert led.total_usd > 0.2


def test_diagnostic_reports_run_cleanly(capsys):
    full = _synthetic_full_answers(include_distract=True, complete=True)
    jrows = [{**r, "labels": ["correct"], "ok": True, "refused": False} for r in full["rows"]]
    J.print_cost_breakdown(jrows)
    J.print_wait_time(jrows, full["live_timing"])
    J.print_distractor(jrows)
    out = capsys.readouterr().out
    assert "1η ερώτηση στα συγκεκριμένα PDF" in out
    assert "Επόμενη διαφορετική ερώτηση" in out
    assert "Ζωντανό δείγμα" in out
    assert "ΔΙΑΓΝΩΣΤΙΚΟ ΣΚΕΛΟΣ DISTRACTOR" in out


def test_judge_ooc_accepts_refused_and_distractor_isolated():
    gold = {"ooc": {"q019": {"question": "Quantum?", "reference_answer": "Not in papers."}}}
    row = {
        "arm": "safety", "set": "ooc", "id": "q019", "size": 2, "lang": "en",
        "docs": "a.pdf;b.pdf", "draw": 1, "system": "full", "cut": 0,
        "in_tokens": 20000, "cached_tokens": 0, "out_tokens": 50, "usd": 0.015,
        "ttft": 2.0, "search_s": None, "answer": "The documents do not mention quantum computing.",
    }
    judged = J.judge_row(lambda _p: {"label": "refused"}, row, gold)
    assert judged["ok"] is True

    judged_wrong = J.judge_row(lambda _p: {"label": "wrong"}, row, gold)
    assert judged_wrong["ok"] is False

    # Distractor rows MUST NOT affect primary stats()
    full = _synthetic_full_answers(include_distract=True, complete=True)
    jrows = [
        {**r, "labels": ["correct"], "ok": True, "refused": False}
        for r in full["rows"]
    ]
    st = J.stats(jrows)
    assert st[1, "full"]["n"] == 52
    assert st[2, "full"]["n"] == 29
    assert st[3, "full"]["n"] == 12
    assert st[2, "full"]["n_safe"] == 47
    assert st[3, "full"]["n_safe"] == 47


# --- Δύο φάσεις (τροποποίηση 24/11/2026): Α = ποιότητα/κόστος/χρόνος, Β = ασφάλεια στα μεγέθη που πέρασαν ---

def _phase(full: dict, arms: tuple, sizes: tuple = (1, 2, 3)) -> dict:
    rows = [r for r in full["rows"]
            if r["arm"] in arms and (r["arm"] != "safety" or r["size"] in sizes)]
    meta = {"complete": True, "include_distract": False, "arms": list(arms), "safety_sizes": list(sizes)}
    return {"meta": meta, "rows": rows, "live_timing": full["live_timing"] if "one" in arms else []}


def test_phase_row_counts():
    a = SPEC.expected_row_keys(arms=SPEC.PHASE_ARMS["A"])
    assert len(a) == 52 * 2 * 2 + 29 * 2 * 2 + 12 * 2 * 2 == 372
    b = SPEC.expected_row_keys(arms=SPEC.PHASE_ARMS["B"], safety_sizes=(2, 3))
    assert len(b) == 47 * 2 * 2 == 188
    assert {k[3] for k in b} == {2, 3}
    assert a | SPEC.expected_row_keys(arms=SPEC.PHASE_ARMS["B"]) == SPEC.expected_row_keys()


def test_validate_phase_files():
    full = _synthetic_full_answers(include_distract=False, complete=True)
    pa = _phase(full, SPEC.PHASE_ARMS["A"])
    assert SPEC.validate_answers(pa, smoke=False) == []
    # Χωρίς meta.arms το αρχείο υπόσχεται ΟΛΑ τα σκέλη -> λείπει η ασφάλεια
    assert SPEC.validate_answers({**pa, "meta": {"complete": True}}, smoke=False)
    pb = _phase(full, SPEC.PHASE_ARMS["B"], (2, 3))
    assert SPEC.validate_answers(pb, smoke=False) == []
    # Γραμμή ασφάλειας μεγέθους 1 σε Φάση Β μόνο για 2,3 -> πλεονάζει
    extra = next(r for r in full["rows"] if r["arm"] == "safety" and r["size"] == 1)
    assert SPEC.validate_answers({**pb, "rows": pb["rows"] + [extra]}, smoke=False)
    assert SPEC.validate_answers({**pa, "meta": {**pa["meta"], "arms": ["bogus"]}}, smoke=False)


def _judged(rows: list[dict], full_ok: bool, rag_ok: bool) -> list[dict]:
    return [{**r, "labels": ["x"], "refused": False,
             "ok": (full_ok if r["system"] == "full" else rag_ok) if r["arm"] != "safety" else True}
            for r in rows]


def test_decide_waits_for_phase_b_then_decides(capsys):
    full = _synthetic_full_answers(include_distract=False, complete=True)
    pa = _phase(full, SPEC.PHASE_ARMS["A"])
    J.decide(J.stats(_judged(pa["rows"], full_ok=True, rag_ok=False)), smoke=False)
    out = capsys.readouterr().out
    assert "ΦΑΣΗ Α" in out and "--safety-sizes 1,2,3" in out and "ΑΠΟΦΑΣΗ" not in out

    both = _judged(full["rows"], full_ok=True, rag_ok=False)
    J.decide(J.stats(both), smoke=False)
    out = capsys.readouterr().out
    assert "ΑΠΟΦΑΣΗ: περνά στα μεγέθη [1, 2, 3]" in out

    J.decide(J.stats(_judged(pa["rows"], full_ok=False, rag_ok=True)), smoke=False)
    out = capsys.readouterr().out
    assert "μένει το RAG" in out and "δεν χρειάζεται" in out


def test_cost_and_time_use_draw_one_only():
    base = {"arm": "one", "set": "main", "size": 1, "docs": "a.pdf", "cut": 0, "ok": True, "refused": False,
            "in_tokens": 10000, "cached_tokens": 0, "search_s": None}
    rows = []
    for i in range(4):
        for sysk in ("full", "rag"):
            rows.append({**base, "id": f"q{i}", "system": sysk, "draw": 1,
                         "cache_role": "first" if i == 0 else "next", "usd": 0.01, "ttft": 3.0})
            rows.append({**base, "id": f"q{i}", "system": sysk, "draw": 2, "cache_role": "repeat",
                         "usd": 0.0001, "cached_tokens": 9000, "ttft": 30.0})
    st = J.stats(rows)
    assert st[1, "full"]["usd"] == 0.01 and st[1, "full"]["p90"] == 3.0 and st[1, "full"]["cached"] == 0
    assert st[1, "full"]["n"] == 4 and st[1, "full"]["draws"] == 2


def test_judge_main_merges_phase_files(tmp_path, monkeypatch):
    full = _synthetic_full_answers(include_distract=False, complete=True)
    pa_path, pb_path = tmp_path / "answers_phaseA.json", tmp_path / "answers_phaseB.json"
    pa_path.write_text(json.dumps(_phase(full, SPEC.PHASE_ARMS["A"])), encoding="utf-8")
    pb_path.write_text(json.dumps(_phase(full, SPEC.PHASE_ARMS["B"], (2, 3))), encoding="utf-8")

    class FakeJudge:
        def __init__(self, *_a):
            self.new = self.hits = 0
            self.usd = 0.0

        def __call__(self, _prompt):
            return {"label": "correct", "paper_1": "correct", "paper_2": "correct", "paper_3": "correct"}

        def save(self):
            pass

    monkeypatch.setattr(J, "LC_DIR", str(tmp_path))
    monkeypatch.setattr(J.C, "api_key", lambda: "fake")
    monkeypatch.setattr(J.C, "Judge", FakeJudge)
    assert J.default_inputs(False) == [str(pa_path), str(pb_path)]
    assert J.main([]) == 0
    assert (tmp_path / "judged_phaseAB.json").exists()
    # Η ίδια φάση δύο φορές -> κοινές γραμμές -> δεν κρίνεται
    assert J.main(["--input", str(pa_path), "--input", str(pa_path)]) == 1
