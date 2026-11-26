"""Βήμα 7 (τελική μέτρηση στο κλειστό σετ): οι κανόνες που κλειδώνονται ΠΡΙΝ από το τρέξιμο.

Χωρίς μοντέλο, χωρίς δίκτυο, χωρίς το κλειστό σετ: fast CI. Ελέγχει τον έλεγχο αριθμών πινάκων (και με
ελληνικούς δεκαδικούς), τους κανόνες «σωστό / όχι», τα υποψήφια κρίσιμα λάθη, το δείγμα του ανθρώπινου
ελέγχου, ότι οι παγωμένες ρυθμίσεις είναι τα defaults του ai_core, και την αναφορά σε συνθετικά δεδομένα.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "evaluation")
sys.path.insert(0, EVAL)

import heldout_common as H  # noqa: E402
import judge_heldout as J  # noqa: E402
import run_heldout as R  # noqa: E402


def item(**kw):
    base = {"id": "x", "type": "single", "lang": "en", "doc1": "a.pdf", "doc2": None, "question": "q",
            "reference": "r", "page1": 3, "quote1": "", "page2": None, "quote2": "", "missing": "",
            "behaviour": "answer", "values": []}
    base.update(kw)
    return base


OK = {"status": "ok", "cited": ["S1"], "outcome": "passed_gate", "sources": [], "passages": {"status": "done"}}


# ------------------------------------------------------------------------------------ τιμές πινάκων

def test_parse_table_values_plain_and_not():
    v = H.parse_table_values("0.95 · accuracy · faithfulness/Ragas; −3 · % · drop; 1e-5 · - · lr; 2-4 · x · y")
    assert [x["value"] for x in v] == [0.95, -3.0, None, None]
    assert v[0]["unit"] == "accuracy" and v[0]["label"] == "faithfulness/Ragas"


def test_greek_decimal_comma_with_three_digits():
    # Ο παλιός έλεγχος διάβαζε το «0,423» ως 423 (κόμμα πριν από 3 ψηφία = χιλιάδες στα αγγλικά).
    assert H.value_found(0.423, H.answer_numbers("η ακρίβεια είναι 0,423 [S2]"))
    assert H.value_found(0.423, H.answer_numbers("precision is 0.423 [S2]"))


def test_thousands_both_conventions():
    assert H.value_found(29423, H.answer_numbers("29,423 requests"))
    assert H.value_found(1000, H.answer_numbers("1.000 κύκλοι"))
    assert H.value_found(29423, H.answer_numbers("29 423 requests"))


def test_whole_token_sign_and_citations():
    nums = H.answer_numbers("10.5 GB and −0.5 dB [S3], 22.0 s")
    assert not H.value_found(0.5, nums)          # το 0.5 δεν είναι μέσα στο 10.5 ούτε στο −0.5
    assert H.value_found(-0.5, nums)
    assert H.value_found(22, nums)
    assert not H.value_found(3, nums)            # το [S3] είναι παραπομπή
    assert not H.value_found(900, H.answer_numbers("9000 seconds"))


def test_range_in_answer_does_not_invent_negative():
    nums = H.answer_numbers("between 0.5-0.7 of the runs")
    assert H.value_found(0.7, nums) and not H.value_found(-0.7, nums)


def test_check_table_values():
    vals = H.parse_table_values("900 · seconds · Max Run Time; 0.5 · GiB · Local Storage; 1e-3 · - · lr")
    res = H.check_table_values(vals, "Μέγιστος χρόνος 900 δευτερόλεπτα και 0,5 GiB [S1].")
    assert res == {"plain_total": 2, "plain_found": 2, "plain_ok": True, "non_plain": 1}
    assert not H.check_table_values(vals, "900 seconds")["plain_ok"]


# ------------------------------------------------------------------------------------- στατιστικά

def test_wilson_matches_known_values():
    lo, hi = H.wilson(10, 12)
    assert round(lo, 3) == 0.552 and round(hi, 3) == 0.953
    lo, hi = H.wilson(6, 8)
    assert round(lo, 2) == 0.41 and round(hi, 2) == 0.93
    assert H.wilson(0, 0) == (None, None)


def test_percentile():
    assert H.percentile([1, 2, 3, 4, 5], 50) == 3
    assert H.percentile([1, 2, 3, 4, 5], 95) == 4.8
    assert H.percentile([], 95) is None


# ------------------------------------------------------------------------------- κανόνες επιτυχίας

def test_single_needs_citation_except_closed_book():
    it = item()
    assert H.decide(it, "alpha", OK, {"main": "correct"})["correct"] is True
    assert H.decide(it, "alpha", {**OK, "cited": []}, {"main": "correct"})["correct"] is False
    assert H.decide(it, "closed", {"status": "ok"}, {"main": "correct"})["correct"] is True
    assert H.decide(it, "alpha", OK, {"main": "partial"})["correct"] is False


def test_technical_failure_is_neither():
    assert H.decide(item(), "alpha", {"status": "technical_failure"}, {})["correct"] is None


def test_two_papers_both_parts():
    it = item(type="two_papers", doc2="b.pdf")
    assert H.decide(it, "beta", OK, {"mh": {"paper_1": "correct", "paper_2": "correct"}})["correct"]
    assert not H.decide(it, "beta", OK, {"mh": {"paper_1": "correct", "paper_2": "partial"}})["correct"]


def test_near_refuse_and_partial():
    ref = item(type="near", behaviour="refuse", page1=None)
    assert H.decide(ref, "alpha", {**OK, "outcome": "cut"}, {})["correct"] is True      # σιωπή φύλακα
    assert H.decide(ref, "alpha", OK, {"near": "correct"})["correct"] is True
    assert H.decide(ref, "alpha", OK, {"near": "soft_leak"})["correct"] is False
    part = item(type="near", behaviour="partial")
    assert H.decide(part, "alpha", OK, {"near": "correct", "main": "correct"})["correct"] is True
    assert H.decide(part, "alpha", OK, {"near": "correct", "main": "partial"})["correct"] is False
    assert H.decide(part, "alpha", {**OK, "outcome": "cut"}, {"near": "correct", "main": "refused"})["correct"] is False


def test_table_needs_value_and_judge():
    it = item(type="table", values=H.parse_table_values("900 · s · run"))
    good = {"main": "correct", "table": {"plain_ok": True, "plain_found": 1, "plain_total": 1}}
    assert H.decide(it, "alpha", OK, good)["correct"] is True
    assert H.decide(it, "alpha", OK, {**good, "main": "partial"})["correct"] is False
    assert H.decide(it, "alpha", OK, {**good, "table": {"plain_ok": False}})["correct"] is False


def test_false_refusal():
    it = item()
    assert H.false_refusal(it, {**OK, "outcome": "cut"}, {})
    assert H.false_refusal(it, OK, {"main": "refused"})
    assert not H.false_refusal(item(type="near", behaviour="refuse"), {**OK, "outcome": "cut"}, {})
    two = item(type="two_papers", doc2="b.pdf")
    assert H.false_refusal(two, OK, {"mh": {"paper_1": "correct", "paper_2": "declared_missing"}})


def test_critical_candidates():
    row = {**OK, "passages": {"status": "done", "items": [{"status": "not_found"}], "numbers_missing": []}}
    assert H.critical_candidates(item(), "alpha", row, {"main": "correct"}) == ["c1"]
    row = {**OK, "passages": {"status": "done", "items": [], "numbers_missing": [{"i": 1, "numbers": ["7"]}]}}
    assert "c1" in H.critical_candidates(item(), "alpha", row, {})
    tab = item(type="table", values=H.parse_table_values("900 · s · run"))
    assert "c2" in H.critical_candidates(tab, "alpha", OK, {"main": "correct", "table": {"plain_ok": False}})
    assert "c2" not in H.critical_candidates(tab, "alpha", OK, {"main": "refused", "table": {"plain_ok": False}})
    ref = item(type="near", behaviour="refuse")
    assert H.critical_candidates(ref, "alpha", OK, {"near": "leak"}) == ["c3"]
    assert H.critical_candidates(ref, "alpha", OK, {"near": "soft_leak"}) == []
    row = {**OK, "sources": [{"label": "S1", "file": "a.pdf", "page": 3}],
           "passages": {"status": "done", "items": [{"status": "found_elsewhere", "file": "b.pdf"}]}}
    assert "c4" in H.critical_candidates(item(), "alpha", row, {})
    two = item(type="two_papers", doc2="b.pdf")
    assert "c4" in H.critical_candidates(two, "alpha", OK, {"mh": {"paper_1": "wrong", "paper_2": "correct"}})
    assert H.critical_candidates(item(), "closed", row, {"main": "wrong"}) == []


def test_review_flags():
    row = {**OK, "other_papers_cited": ["z.pdf"], "passages": {"status": "failed"}}
    tab = item(type="table", values=H.parse_table_values("1e-3 · - · lr"))
    assert H.review_flags(tab, "alpha", row) == ["other_paper_cited", "non_plain_value", "passage_check_missing"]
    assert H.review_flags(tab, "alpha", row, False)[0] == "other_paper_cited"
    # σωστή κατά τον κριτή: η παραπομπή σε άλλο άρθρο δεν χρειάζεται έλεγχο λάθους του σετ
    assert "other_paper_cited" not in H.review_flags(tab, "alpha", row, True)
    assert H.review_flags(item(), "alpha", OK) == []


def test_review_sample_is_deterministic_and_per_category():
    pool = {c: [f"{c}:{i}" for i in range(6)] for c in H.CATEGORIES}
    a, b = H.sample_random(pool), H.sample_random(pool)
    assert a == b and len(a) == 2 * len(H.CATEGORIES)
    for c in H.CATEGORIES:
        assert sum(k.startswith(c + ":") for k in a) == 2
    assert H.sample_random({"near": ["near:1"]}) == ["near:1"]   # λιγότερα από 2 -> όσα υπάρχουν


# ---------------------------------------------------------------------------------------- αρχεία

def test_load_items_greek_headers(tmp_path):
    path = tmp_path / "q.csv"
    head = ["id", "τύπος", "γλώσσα", "άρθρο_1", "άρθρο_2", "ερώτηση", "απάντηση_αναφοράς", "σελίδα_1",
            "απόσπασμα_1", "σελίδα_2", "απόσπασμα_2", "τι_λείπει", "σωστή_συμπεριφορά", "τιμές_πίνακα"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerow(["h1", "two_papers", "el", "a.pdf", "b", "ερ", "απ", "3", "αα", "7", "ββ", "", "answer", ""])
    it = H.load_items(str(path))[0]
    assert (it["doc1"], it["doc2"], it["page1"], it["page2"]) == ("a.pdf", "b.pdf", 3, 7)


def test_system_files_exclude_evaluation_and_tests():
    assert H.is_system_file("backend/ai_core.py")
    assert H.is_system_file("backend/corpus_names.json")
    assert not H.is_system_file("backend/evaluation/run_heldout.py")
    assert not H.is_system_file("backend/tests/test_heldout.py")
    assert not H.is_system_file("frontend/app_ui.py")


def test_frozen_config_matches_ai_core_defaults():
    """Οι παγωμένες τιμές = τα defaults στον κώδικα του ai_core (το run_heldout ελέγχει τις ΠΡΑΓΜΑΤΙΚΕΣ)."""
    with open(os.path.join(HERE, "..", "ai_core.py"), encoding="utf-8") as f:
        src = f.read()

    def default(name):
        m = re.search(rf'os\.getenv\("{name}", "([^"]*)"\)', src)
        assert m, name
        return m.group(1)

    fc = H.FROZEN_CONFIG
    assert default("GEMINI_MODEL") == fc["GEMINI_MODEL"]
    assert default("RERANKER_MODEL") == fc["RERANKER_MODEL"]
    assert float(default("MIN_RERANK_SCORE")) == fc["MIN_RERANK_SCORE"]
    assert float(default("CORRECTIVE_MIN_SCORE")) == fc["CORRECTIVE_MIN_SCORE"]
    assert int(default("THINKING_BUDGET")) == fc["THINKING_BUDGET"]
    for name in ("RERANK_CANDIDATES", "DENSE_CANDIDATES", "EXPAND_INPUT", "MAX_PAGES", "PAGE_SCORE_TOP_K",
                 "PERDOC_EXTRA"):
        assert int(default(name)) == fc[name], name
    assert default("STRIP_HIDDEN_TEXT") == "0" and fc["STRIP_HIDDEN_TEXT"] is False
    assert default("USE_BGE_SPARSE") == "0" and fc["USE_BGE_SPARSE"] is False
    with open(os.path.join(HERE, "..", "full_text.py"), encoding="utf-8") as f:
        assert re.search(r"^MAX_TOKENS = 70_000", f.read(), re.M) and fc["FULL_TEXT_MAX_TOKENS"] == 70000


# ------------------------------------------------------------------------------- runner και κριτής

def test_runner_targets_mirror_the_ui():
    store = {"doc_ids": {"a.pdf": 7000, "b.pdf": 7001, "c.pdf": 7002}}
    it = item(type="two_papers", doc1="c.pdf", doc2="a.pdf")
    assert R.targets(it, "alpha", store) == (["a.pdf", "b.pdf", "c.pdf"], [7000, 7001, 7002])
    assert R.targets(it, "beta", store) == (["c.pdf", "a.pdf"], [7002, 7000])


def test_runner_evidence_stages():
    store = {"text_pages": {"chunk-a3": {("a.pdf", 3)}, "chunk-b7": {("b.pdf", 7)}}}
    R.SPY.reset()
    R.SPY.rerank.append({"max": 1.0, "texts": ["chunk-a3"]})
    R.SPY.rerank.append({"max": -3.0, "texts": ["chunk-b7"]})
    it = item(type="two_papers", doc2="b.pdf", page2=7)
    sources = [{"label": "S1", "file": "a.pdf", "page": 3}]
    ev = R.evidence(it, sources, ["S1"], store)
    assert ev[0] == {"file": "a.pdf", "page": 3, "in_candidates_first": True, "in_candidates_any": True,
                     "in_final": True, "cited": True}
    assert ev[1]["in_candidates_first"] is False and ev[1]["in_candidates_any"] is True
    assert ev[1]["in_final"] is False
    R.SPY.reset()
    assert R.evidence(item(page1=None), [], [], store) == [None]


def test_runner_aux_kinds_and_cost():
    assert R._aux_kind("You are a translation assistant ...") == "translation"
    assert R._aux_kind("The following search query ...") == "corrective"
    assert R._aux_kind(R.CLOSED_PROMPT.format(question="q")) == "closed_book"
    usd = R.est_usd({"prompt_tokens": 1_000_000, "completion_tokens": 0, "aux": []}, 0.3, 2.5)
    assert abs(usd - 0.3) < 1e-9


def test_judge_mh_prompt_has_labels_and_formats():
    assert "{labels}" in J.MH_HELDOUT and "Key terms expected" not in J.MH_HELDOUT
    it = item(type="two_papers", doc1="a.pdf", doc2="b.pdf")
    row = {"sources": [{"label": "S1", "file": "a.pdf", "page": 2}, {"label": "S2", "file": "z.pdf", "page": 9}],
           "cited": ["S1", "S2"]}
    labels = J.label_list(it, row, {"z.pdf": "Other"})
    assert labels == "S1 = PAPER 1, page 2; S2 = another paper (Other), page 9"
    J.MH_HELDOUT.format(name1="A", name2="B", question="q", reference="r", kw1="x", kw2="y", labels=labels,
                        answer="{not a placeholder}")


def test_human_judgments_use_the_same_rules():
    tab = item(type="table", values=H.parse_table_values("900 · s · run"))
    h = J.human_judgments(tab, {"verdict": "correct", "wrong_value": "no"})
    assert H.decide(tab, "alpha", OK, h)["correct"] is True
    h = J.human_judgments(tab, {"verdict": "correct", "wrong_value": "yes"})
    assert H.decide(tab, "alpha", OK, h)["correct"] is False
    part = item(type="near", behaviour="partial")
    assert H.decide(part, "alpha", OK, J.human_judgments(part, {"verdict": "correct"}))["correct"] is True
    ref = item(type="near", behaviour="refuse")
    assert H.decide(ref, "alpha", OK, J.human_judgments(ref, {"verdict": "leak"}))["correct"] is False


def test_report_on_synthetic_run():
    items = {"s1": item(id="s1"), "m1": item(id="m1", type="two_papers", doc2="b.pdf", page2=4),
             "n1": item(id="n1", type="near", behaviour="refuse", page1=None)}
    base = {**OK, "sources": [{"label": "S1", "file": "a.pdf", "page": 3}], "evidence": [None],
            "passages": {"status": "done", "items": [{"status": "found"}], "numbers_missing": []},
            "wall_s": 2.0, "generation_s": 1.0, "est_usd": 0.001, "full_text": False}
    rows = {
        "alpha:s1": {**base, "arm": "alpha", "id": "s1"},
        "alpha:m1": {**base, "arm": "alpha", "id": "m1"},
        "alpha:n1": {**base, "arm": "alpha", "id": "n1", "outcome": "cut", "sources": []},
        "beta:m1": {**base, "arm": "beta", "id": "m1", "outcome": "full_text", "full_text": True},
        "closed:s1": {"status": "ok", "arm": "closed", "id": "s1", "outcome": "closed_book", "cited": [],
                      "wall_s": 1.0, "est_usd": 0.0},
        "closed:m1": {"status": "technical_failure", "arm": "closed", "id": "m1", "wall_s": 1.0},
    }
    judged = {"alpha:s1": {"j": {"main": "correct"}},
              "alpha:m1": {"j": {"mh": {"paper_1": "correct", "paper_2": "omitted"}}},
              "alpha:n1": {"j": {"near": "correct"}},
              "beta:m1": {"j": {"mh": {"paper_1": "correct", "paper_2": "correct"}}},
              "closed:s1": {"j": {"main": "wrong"}}}
    rep, pub = J.build_report(items, {"meta": {}, "rows": rows}, judged, {"a.pdf": "rag"}, "smoke", set())
    assert rep["headline_alpha"]["correct"] == 2 and rep["headline_alpha"]["n"] == 3
    assert rep["beta_vs_alpha"]["improved"] == ["m1"] and rep["beta_vs_alpha"]["beta_full_text"] == 1
    assert rep["closed_book"]["two_papers"]["technical"] == 1
    assert rep["closed_vs_alpha"]["only_rag"] == 1
    assert rep["citations"]["sentences"] == 4 and rep["citations"]["with_citation"] == 4
    assert all("answer" not in r and "question" not in r for r in pub)   # η αναφορά δεν έχει κείμενα
