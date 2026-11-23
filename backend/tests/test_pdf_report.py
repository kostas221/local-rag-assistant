"""Αναφορά εξαγωγής κειμένου (pdf_report): «Text extracted from X/Y pages». Χωρίς μοντέλο ούτε pymupdf
-> fast CI. Η ανάγνωση πραγματικού PDF δοκιμάζεται στο test_ingestion.py (core-tests)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pdf_report as R


def test_summarize_counts_pages_and_flags_image_only_pages():
    rep = R.summarize(["Intro text", "   \n", "", "Results 0.5 s"], image_pages=[3])
    assert rep == {"pages_total": 4, "pages_text": 2, "pages_empty": [2, 3], "pages_scanned": [3]}


def test_describe_lists_empty_pages_and_truncates():
    assert R.describe(4, 4, []) == "Text extracted from 4/4 pages"
    assert R.describe(4, 2, [2, 3]) == "Text extracted from 2/4 pages · no text on p. 2, 3"
    many = list(range(1, 13))
    assert R.describe(20, 8, many, max_listed=3).endswith("p. 1, 2, 3 and 9 more")
    assert R.describe(None, None) == ""          # έγγραφα πριν από τις στήλες: καμία ένδειξη


def test_pages_round_trip_through_one_column():
    assert R.pages_from_str(R.pages_to_str([4, 7])) == [4, 7]
    assert R.pages_from_str(None) == [] and R.pages_from_str("") == []
