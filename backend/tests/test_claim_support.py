"""Στήριξη ανά ισχυρισμό (claim_support.py): ντετερμινιστικό μέρος, χωρίς μοντέλο -> fast CI.

Κλειδώνουμε: (1) ο χωρισμός σε προτάσεις κρατά τις ετικέτες, και οι αρνήσεις χωρίς παραπομπή δεν
είναι ισχυρισμοί· (2) το αυτούσιο χωρίο αντέχει παύλες, κεφαλαία και σπασμένες λέξεις αλλά όχι
παράφραση· (3) αλλοιωμένος αριθμός σημαδεύεται ΑΚΟΜΑ κι αν το μοντέλο πει «full»· (4) οι ετικέτες
αριθμούνται όπως στο ask_ai.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import claim_support as CS  # noqa: E402

REFUSAL = re.compile(r"do(?:es)? not (?:mention|provide|report)|δεν αναφέρ")

PAGE_1 = ("Cloud Computing refers to both the applications delivered as services over the "
          "Internet and the hardware and systems software in the data-\ncenters that provide "
          "those services. Twenty-two studies were included.")
PAGE_2 = "The median cold start was 0.5 seconds across 1,346 invocations in 2019."
PAGES = [(PAGE_1, {"doc_id": "a", "page": 1}), (PAGE_2, {"doc_id": "b", "page": 4}),
         (PAGE_1, {"doc_id": "a", "page": 1})]


def test_split_keeps_labels_and_strips_citations():
    ans = ("Cloud Computing covers applications and datacenter hardware [S1, S3]. "
           "The median cold start was 0.5 seconds [S2].")
    cl = CS.split_claims(ans, REFUSAL)
    assert [c.labels for c in cl] == [["S1", "S3"], ["S2"]]
    assert cl[1].text == "The median cold start was 0.5 seconds."
    assert all(c.kind == "claim" for c in cl)


def test_refusal_without_label_is_not_a_claim_but_with_label_it_is():
    ans = ("The documents do not mention the price of Lambda.\n"
           "The paper does not report latency, but it uses 1,346 invocations [S2].")
    cl = CS.split_claims(ans, REFUSAL)
    assert [c.kind for c in cl] == ["refusal", "claim"]


def test_bullets_lead_in_and_trailing_citation():
    ans = ("Τα βασικά σημεία είναι τα εξής:\n- Η καθυστέρηση ήταν 0,5 δευτερόλεπτα. [S2]\n"
           "2. Συμπεριλήφθηκαν δύο μελέτες στο σύνολο [S1].")
    cl = CS.split_claims(ans, REFUSAL)
    assert [c.kind for c in cl] == ["lead_in", "claim", "claim"]
    assert cl[1].labels == ["S2"] and cl[1].text.startswith("Η καθυστέρηση")


def test_quote_in_source_tolerates_layout_but_not_paraphrase():
    assert CS.quote_in_source("hardware and systems software in the datacenters", PAGE_1)
    assert CS.quote_in_source("CLOUD COMPUTING refers to both the applications", PAGE_1)
    assert CS.quote_in_source("cold start was 0.5 seconds", PAGE_2.replace("-", "–"))
    assert not CS.quote_in_source("hardware and software in the datacenters", PAGE_1)
    assert not CS.quote_in_source("0.5", PAGE_2)        # πολύ μικρό για να δείχνει σημείο


def test_numbers_accept_decimal_comma_words_and_question():
    assert CS.unsupported_numbers("Η καθυστέρηση ήταν 0,5 δευτερόλεπτα", [PAGE_2]) == []
    assert CS.unsupported_numbers("22 μελέτες", [PAGE_1]) == []
    assert CS.unsupported_numbers("1346 κλήσεις", [PAGE_2]) == []
    assert CS.unsupported_numbers("0.7 seconds in 2021", [PAGE_2]) == ["0.7", "2021"]
    assert CS.unsupported_numbers("the 2017 paper", [PAGE_2], CS.numbers_in("Baldini 2017")) == []


def test_label_texts_number_like_ask_ai():
    texts = CS.label_texts(PAGES)
    assert list(texts) == ["S1", "S2"]          # η 3η σελίδα είναι ίδια με την 1η -> S1
    assert texts["S1"].count("Cloud Computing refers") == 2


def _check(ans, verdicts):
    texts = CS.label_texts(PAGES)
    cl = CS.split_claims(ans, REFUSAL)
    return CS.apply_check(cl, texts, verdicts)


def test_supported_wrong_spot_partial_and_fake_quote():
    ans = ("Cloud Computing covers applications and datacenter hardware [S1]. "
           "The median cold start was 0.5 seconds [S1]. "
           "The cold start depends on the language runtime [S2]. "
           "There were 1,346 invocations in total [S2].")
    v = {1: {"quote": "Cloud Computing refers to both the applications", "label": "S1",
             "support": "full", "missing": ""},
         2: {"quote": "The median cold start was 0.5 seconds", "label": "S1",   # λάθος ετικέτα
             "support": "full", "missing": ""},
         3: {"quote": "The median cold start was 0.5 seconds", "label": "S2",
             "support": "partial", "missing": "language runtime"},
         4: {"quote": "there were 1346 calls", "label": "S2", "support": "full", "missing": ""}}
    c1, c2, c3, c4 = _check(ans, v)
    assert (c1.status, c1.cite_ok) == ("supported", True)
    assert c2.quote_label == "S2" and c2.quote_found and not c2.cite_ok   # (β) λάθος σημείο
    assert c2.status == "flagged" and c2.reason == "number_missing"     # 0.5 δεν είναι στη S1
    assert (c3.status, c3.reason) == ("flagged", "partial")
    assert (c4.status, c4.reason) == ("flagged", "quote_not_in_source")


def test_mutated_number_is_flagged_even_if_model_says_full():
    ans = "The median cold start was 0.8 seconds [S2]."
    v = {1: {"quote": "The median cold start was 0.5 seconds", "label": "S2", "support": "full",
             "missing": ""}}
    (c,) = _check(ans, v)
    assert (c.status, c.reason, c.numbers_missing) == ("flagged", "number_missing", ["0.8"])


def test_parse_check_tolerates_fences_and_rejects_garbage():
    raw = '```json\n{"claims": [{"i": "2", "quote": "x y", "label": "[S3]", "support": "FULL"}]}\n```'
    assert CS.parse_check(raw) == {2: {"quote": "x y", "label": "S3", "support": "full", "missing": ""}}
    assert CS.parse_check("not json") == {}
    assert CS.parse_check('{"claims": [{"i": null}, 5]}') == {}


def test_no_prompt_when_only_refusals():
    cl = CS.split_claims("The documents do not mention the price of Lambda.", REFUSAL)
    assert CS.build_check_prompt("q", cl, CS.label_texts(PAGES)) is None
    (c,) = CS.apply_check(cl, CS.label_texts(PAGES), None)
    assert (c.status, c.reason) == ("skipped", "refusal")
