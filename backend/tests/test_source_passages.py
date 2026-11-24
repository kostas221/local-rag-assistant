"""Απόσπασμα πηγής ανά πρόταση (κύκλος 2, βήμα 6). Χωρίς μοντέλα -> γρήγορα tests του CI.

Προστατεύει: (1) τις έξι καταστάσεις που βλέπει ο χρήστης, ιδίως ότι ΔΕΝ υπάρχει «επαληθευμένο» και ότι η
κρίση full/partial/none του βήματος 5 δεν φτάνει στο UI· (2) ότι ο έλεγχος παίρνει μόνο παρατεθειμένες
σελίδες και αφήνει τις προτάσεις χωρίς παραπομπή· (3) ότι το regex άρνησης είναι ΙΔΙΟ με αυτό που μετρήθηκε
στο βήμα 5 (eval_near_ooc.REFUSAL)."""
import os
import sys

import claim_support as CS
import source_passages as SP

PAGES = {"S1": "Each Lambda function can run for at most 300 seconds before it is terminated.",
         "S2": "ExCamera workers are limited to five minutes of execution time."}
SOURCES = {"S1": {"label": "S1", "file": "pywren.pdf", "page": 3},
           "S2": {"label": "S2", "file": "excamera.pdf", "page": 4}}


def _run(answer, verdicts):
    claims, prompt = SP.prepare("what limits?", answer, PAGES)
    claims = CS.apply_check(claims, PAGES, verdicts, "what limits?")
    return prompt, SP.to_items(claims, SOURCES)


def test_found_on_the_cited_page():
    answer = "PyWren reports a maximum of 300 seconds per function [S1]."
    prompt, items = _run(answer, {1: {"quote": "can run for at most 300 seconds", "label": "S1",
                                      "support": "partial", "missing": "x"}})
    assert prompt and "[S1]" in prompt
    it = items[0]
    assert it["status"] == SP.FOUND and it["quote"] == "can run for at most 300 seconds"
    assert (it["file"], it["page"]) == ("pywren.pdf", 3)
    assert "support" not in it and "missing" not in it        # η κρίση του μοντέλου δεν φτάνει στο UI


def test_found_on_another_page_than_cited():
    answer = "ExCamera workers run for at most five minutes [S1]."
    _p, items = _run(answer, {1: {"quote": "limited to five minutes of execution time", "label": "S1",
                                  "support": "full", "missing": ""}})
    assert items[0]["status"] == SP.FOUND_ELSEWHERE and items[0]["label"] == "S2"


def test_quote_not_verbatim_is_not_found():
    answer = "PyWren functions can run for ten minutes [S1]."
    _p, items = _run(answer, {1: {"quote": "can run for ten minutes", "label": "S1",
                                  "support": "full", "missing": ""}})
    assert items[0]["status"] == SP.NOT_FOUND and items[0]["quote"] is None


def test_uncited_and_absence_sentences_are_not_checked():
    answer = ("PyWren reports a maximum of 300 seconds per function [S1].\n"
              "This limit shapes how the system schedules all of its work.\n"
              "The documents do not mention any GPU support [S2].")
    prompt, items = _run(answer, {1: {"quote": "can run for at most 300 seconds", "label": "S1",
                                      "support": "full", "missing": ""}})
    assert prompt.count("(cites") == 2                          # η πρόταση χωρίς παραπομπή λείπει
    assert [it["status"] for it in items] == [SP.FOUND, SP.NO_CITATION, SP.ABSENCE]


def test_no_cited_sentence_means_no_call():
    _claims, prompt = SP.prepare("q", "A sentence without any citation at all.", PAGES)
    assert prompt is None


def test_cited_labels_in_order_of_first_use():
    assert SP.cited_labels("A [S3]. B [S1, S3]. C [S2][S1].") == ["S3", "S1", "S2"]


def test_refusal_regex_is_the_measured_one():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "evaluation"))
    import eval_near_ooc as E
    assert SP.REFUSAL.pattern == E.REFUSAL.pattern and SP.REFUSAL.flags == E.REFUSAL.flags
