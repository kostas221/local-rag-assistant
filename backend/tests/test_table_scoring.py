"""Αυστηρός έλεγχος απαντήσεων πινάκων (evaluation/table_scoring.py). Χωρίς μοντέλο -> fast CI.

Ο παλιός έλεγχος (εμφάνιση κειμένου) έδινε 2/2 και σε λάθος απαντήσεις της t001. Εδώ κλειδώνουμε:
(1) ότι ο αυστηρός πιάνει υποσυμβολοσειρά, αντιμετάθεση, λάθος μονάδα και πρόσημο, (2) ότι δέχεται
δεκαδικό κόμμα και διαφορετική σειρά λέξεων, και (3) ότι οι 16 αποθηκευμένες απαντήσεις της baseline,
ελεγμένες με το χέρι απέναντι στα PDF (18/11/2026), περνούν όλες.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "evaluation")
sys.path.insert(0, EVAL)

import table_scoring as TS  # noqa: E402

with open(os.path.join(EVAL, "golden_tables.jsonl"), encoding="utf-8") as _f:
    GOLDEN = {q["id"]: q for q in map(json.loads, _f)}


def facts(qid):
    return GOLDEN[qid]["facts"]


def test_t001_substring_and_swap_are_caught():
    f = facts("t001")
    assert TS.check_strict("900 seconds and 0.5 GB", f) == [True, True]
    assert TS.check_strict("9000 seconds and 10.5 GB", f) == [False, False]
    assert TS.check_strict("0.5 seconds and 900 GB", f) == [False, False]


def test_decimal_comma_and_thousands_separator():
    assert TS.check_strict("Ο χρόνος είναι 900 seconds και ο χώρος 0,5 GiB", facts("t001")) == [True, True]
    assert TS.check_strict("29,423 jobs, average completion time 634 secs", facts("t005")) == [True, True]
    out = TS.check_strict("Storage: medium-sized datacenter $2,20 per GByte/month, very large datacenter "
                          "$0,40 per GByte/month. Network: medium $95 per Mbit/sec/month, very large "
                          "$13 per Mbit/sec/month.", facts("t011"))
    assert out == [True, True, True, True]


def test_swapped_columns_with_same_unit_are_caught():
    swapped = ("The minimum accounting unit for serverless cloud functions is 60 seconds, whereas for "
               "serverful VMs it is 0.1 seconds.")
    assert TS.check_strict(swapped, facts("t002")) == [False, False]
    assert TS.check_strict("1. Lambda start latency: mean 29.1 s and standard deviation 9.7 s",
                           facts("t015"))[:2] == [False, False]


def test_label_after_number_and_shared_and():
    # Η ετικέτα ΜΕΤΑ τον αριθμό, και «and» ανάμεσα σε δύο τιμές (naive_tables t003, σωστή απάντηση).
    ok = ("According to a 2018 survey, web and API serving accounts for 32% and data processing, "
          "such as batch ETL, accounts for 21%.")
    assert TS.check_strict(ok, facts("t003")) == [True, True]
    assert TS.check_strict("32% is attributed to web and API serving, and 21% to data processing.",
                           facts("t003")) == [True, True]


def test_wrong_unit_and_sign():
    f = [{"value": 5, "unit": [r"mg\b"]}]
    assert TS.check_strict("a dose of 5 mg", f) == [True]
    assert TS.check_strict("a dose of 5 g", f) == [False]
    g = [{"value": -20, "unit": [r"s\b"]}]
    assert TS.check_strict("the time changed by −20 s", g) == [True]
    assert TS.check_strict("the time changed by +20 s", g) == [False]
    assert TS.check_strict("the time changed by 20 s", g) == [False]


def test_every_golden_question_has_valid_facts():
    for qid, q in GOLDEN.items():
        assert q.get("facts"), qid
        for f in q["facts"]:
            assert ("value" in f) != ("text" in f), (qid, f)
            for p in [f.get("text"), f.get("anchor"), f.get("row"), *f.get("unit", [])]:
                if p:
                    re.compile(p)


def test_stored_baseline_answers_all_pass():
    path = os.path.join(EVAL, "runs", "scoreboard", "answers_baseline_tables.csv")
    with open(path, encoding="utf-8-sig") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    for qid, q in GOLDEN.items():
        assert all(TS.check_strict(rows[qid]["answer"], q["facts"])), qid
