"""Σελίδα PDF με το χωρίο υπογραμμισμένο (page_view.py, 26/11/2026).

Ο κανόνας που κλειδώνεται: κίτρινο ΜΟΝΟ όταν κάθε κομμάτι του χωρίου υπάρχει ακριβώς μία φορά στη σελίδα, με την
ίδια κανονικοποίηση με τον έλεγχο χωρίων. Λάθος υπογράμμιση είναι χειρότερη από καμία. Τα καθαρά tests τρέχουν
χωρίς pymupdf (fast CI)· η εικόνα μόνο όπου υπάρχει pymupdf.

Μέτρηση στα χωρία ανάπτυξης του βήματος 5 (488 μοναδικά, κλήρωση 1): 481 μοναδικό σημείο, 3 αμφίσημα, 4 δεν
εντοπίστηκαν· πίνακες 48/48. Οπτικός έλεγχος 7 εικόνων: το κίτρινο στο σωστό σημείο και στις 7.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import page_view as PV


def test_canonical_words_join_hyphenated_and_keep_numbers_apart():
    text, spans = PV.canonical_words(["distribu-", "ted", "systems", "9.1", "501.13"])
    assert text == "distributedsystems9.1|501.13"
    # η κομμένη λέξη και η συνέχειά της δείχνουν σε δύο λέξεις της σελίδας
    assert [i for _s, _e, i in spans] == [0, 1, 2, 3, 4]


def test_unique_passage_is_highlighted_across_a_line_break():
    words = ["In", "this", "paper", "we", "show", "that", "serverless", "computing", "is", "the", "future", "of", "cloud", "com-", "puting", "."]
    status, idxs = PV.locate("serverless computing is the future of cloud computing", words)
    assert status == "highlighted"
    assert [words[i] for i in idxs] == ["serverless", "computing", "is", "the", "future", "of", "cloud",
                                        "com-", "puting"]


def test_repeated_text_is_not_highlighted():
    words = ["Latency", "12", "hours", "row", "A", "and", "latency", "12", "hours", "row", "B"]
    assert PV.locate("latency 12 hours", words) == ("ambiguous", [])


def test_table_cells_highlight_only_when_each_cell_is_unique():
    words = ["Minimum", "Accounting", "Unit", "0.1", "seconds", "60", "seconds", "Maximum", "Run", "Time", "900", "seconds"]
    status, idxs = PV.locate("Minimum Accounting Unit\n0.1 seconds", words)
    assert status == "highlighted" and [words[i] for i in idxs][:3] == ["Minimum", "Accounting", "Unit"]
    # «seconds» μόνο του υπάρχει τρεις φορές: χωρισμένο κελί -> αμφίσημο, καθόλου κίτρινο
    assert PV.locate("Maximum Run Time ... seconds", words)[0] == "ambiguous"


def test_number_must_match_whole():
    # το «20» δεν είναι μέσα στο «120», ούτε το «0.5» μέσα στο «10.5»
    assert PV.locate("20 ms", ["a", "delay", "of", "120", "ms"])[0] == "not_found"
    assert PV.locate("0.5 GiB", ["uses", "10.5", "GiB"])[0] == "not_found"
    assert PV.locate("0.5 GiB", ["uses", "0.5", "GiB"])[0] == "highlighted"


def test_missing_passage_is_reported():
    assert PV.locate("something the page never says", ["a", "page", "about", "clouds"]) == ("not_found", [])


def test_render_png_with_highlight(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = str(tmp_path / "doc.pdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Serverless computing scales to zero when idle.")
    page.insert_text((72, 100), "Another line about cold starts.")
    doc.save(path)
    doc.close()

    png, status = PV.render(path, 1, "scales to zero when idle", zoom=1.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and status == "highlighted"
    assert PV.render(path, 1)[1] == "none"
    assert PV.render(path, 1, "not on this page at all")[1] == "not_found"
    with pytest.raises(PV.PageNotFound):
        PV.render(path, 2)
    # το αρχείο δεν αλλάζει: η υπογράμμιση μπαίνει μόνο στη μνήμη
    with pymupdf.open(path) as d:
        assert len(list(d[0].annots())) == 0
