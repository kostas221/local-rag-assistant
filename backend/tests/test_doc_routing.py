"""Δρομολόγηση ερωτήσεων δύο εγγράφων (doc_routing, leaf module -> ΚΑΝΕΝΑ μοντέλο, fast CI).

ΤΙ ΕΛΕΓΧΕΤΑΙ ΕΔΩ: η απόφαση «ονομάζει η ερώτηση ≥ 2 έγγραφα;», η σειρά τους, και ποιες σελίδες
προστίθενται. ΚΑΙ ότι τα ονόματα του corpus_names.json είναι ΑΚΡΙΒΩΣ αυτά που μετρήθηκαν
(evaluation/scoreboard_answers.ALIASES): μια αλλαγή στη λίστα αλλάζει ποιες ερωτήσεις δρομολογούνται,
άρα θέλει νέα μέτρηση. Η σύνδεση με το search_documents ελέγχεται άκρο-έως-άκρο από το
evaluation/scoreboard.py με ENABLE_PERDOC=1.
"""
import ast
import os

import doc_routing

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES_PATH = os.path.join(HERE, "..", "corpus_names.json")
ALL = {"1702.04024.pdf", "1706.03178.pdf", "1812.03651.pdf", "1902.03383v1.pdf",
       "EECS-2009-28.pdf", "excamera-nsdi17.pdf", "mapreduce-osdi04.pdf"}


def _names():
    return doc_routing.load_names(NAMES_PATH)


def _page(doc, page):
    return (f"text {doc}:{page}", {"file_name": doc, "page": page})


def test_names_are_exactly_the_measured_aliases():
    with open(os.path.join(HERE, "..", "evaluation", "scoreboard_answers.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    measured = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ALIASES")
    assert _names() == {doc: list(al) for doc, al in measured.items()}


def test_two_named_docs_in_mention_order():
    q = "What does MapReduce do about stragglers, and how does PyWren handle them?"
    assert doc_routing.named_docs(q, _names(), ALL) == ["mapreduce-osdi04.pdf", "1702.04024.pdf"]


def test_greek_question_with_latin_names():
    q = "Πώς περιγράφουν το κόστος το PyWren και το 'Above the Clouds';"
    assert doc_routing.named_docs(q, _names(), ALL) == ["1702.04024.pdf", "EECS-2009-28.pdf"]


def test_single_doc_is_not_routed():
    assert doc_routing.named_docs("What is PyWren?", _names(), ALL) == ["1702.04024.pdf"]


def test_out_of_scope_doc_is_ignored():
    q = "Compare PyWren and MapReduce."
    assert doc_routing.named_docs(q, _names(), {"mapreduce-osdi04.pdf"}) == ["mapreduce-osdi04.pdf"]


def test_missing_or_broken_names_file_disables_routing(tmp_path):
    assert doc_routing.load_names(str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert doc_routing.load_names(str(bad)) == {}


def test_extra_pages_skip_base_alternate_and_limit():
    base = [_page("a.pdf", 1), _page("b.pdf", 1)]
    legs = [[_page("a.pdf", 1), _page("a.pdf", 2), _page("a.pdf", 3)],
            [_page("b.pdf", 1), _page("b.pdf", 2), _page("b.pdf", 3)]]
    got = doc_routing.extra_pages(base, legs, 3)
    assert [m["page"] for _t, m in got] == [2, 2, 3]
    assert [m["file_name"] for _t, m in got] == ["a.pdf", "b.pdf", "a.pdf"]


def test_extra_pages_zero_limit_and_no_duplicates():
    base = [_page("a.pdf", 1)]
    assert doc_routing.extra_pages(base, [[_page("a.pdf", 2)]], 0) == []
    got = doc_routing.extra_pages(base, [[_page("a.pdf", 2)], [_page("a.pdf", 2)]], 4)
    assert len(got) == 1
