"""Ο καθολικός descriptor ισχύει ΜΟΝΟ για τα αρχεία που περιγράφει (28/9/2026).

ΤΟ ΣΦΑΛΜΑ: το ingest γράφει domain="" (στατιστικό γλωσσάρι) -> το optimize_query έπεφτε στο
corpus_descriptor.json για ΚΑΘΕ αρχείο. Ένα ανεβασμένο PDF καρδιολογίας μεταφραζόταν με «The corpus
is about: Cloud Computing and Distributed Systems», και η «Πότε εξερράγη ο Βεζούβιος;» περνούσε τον
φύλακα 5/5 πάνω σε σελίδες για το Στρόμπολι· με γενικό πεδίο 1/5 (evaluation/probe_vesuvius_domain.py).

ΤΙ ΕΛΕΓΧΕΤΑΙ ΕΔΩ: η απόφαση «έχει το scope ΕΣΤΩ ΕΝΑ αρχείο του descriptor;» (corpus_glossary,
leaf module -> ΚΑΝΕΝΑ μοντέλο, fast CI). Η σύνδεση με το search_documents ελέγχεται άκρο-έως-άκρο
από το evaluation/scoreboard.py (γραμμή «άλλα πεδία · πεδίο στο prompt μετάφρασης»).
"""
import json
import os

import corpus_glossary

DESCRIPTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus_descriptor.json")
CLOUD = {"1702.04024.pdf", "excamera-nsdi17.pdf"}


def _idx(files: list[str]) -> tuple[list[str], dict]:
    """Ίδιο σχήμα με το BM25 cache του ai_core: ids, metas ευθυγραμμισμένα, pos = {id: θέση}."""
    ids = [f"c{i}" for i in range(len(files))]
    metas = [{"file_name": f} for f in files]
    return ids, {"ids": ids, "metas": metas, "pos": {cid: i for i, cid in enumerate(ids)}}


def test_scope_with_only_descriptor_files():
    ids, idx = _idx(["1702.04024.pdf", "excamera-nsdi17.pdf"])
    assert corpus_glossary.scope_has_files(ids, idx, CLOUD) is True


def test_scope_without_descriptor_files_is_the_bug_case():
    # PDF καρδιολογίας/ηφαιστειολογίας: ΚΑΝΕΝΑ αρχείο του descriptor -> γενικό πεδίο
    ids, idx = _idx(["cureus-0015-00000046486.pdf", "s41598-017-03833-3.pdf"])
    assert corpus_glossary.scope_has_files(ids, idx, CLOUD) is False


def test_mixed_scope_keeps_descriptor():
    # Μικτό scope: συμπεριφορά ΟΠΩΣ ΠΡΙΝ (δεν μετρήθηκε — δεν αλλάζει χωρίς μέτρηση)
    ids, idx = _idx(["cureus-0015-00000046486.pdf", "excamera-nsdi17.pdf"])
    assert corpus_glossary.scope_has_files(ids, idx, CLOUD) is True


def test_only_allowed_ids_count():
    # Τα ids εκτός authz/επιλογής αρχείων ΔΕΝ μετράνε, ακόμα κι αν το αρχείο τους είναι του descriptor
    ids, idx = _idx(["cureus-0015-00000046486.pdf", "excamera-nsdi17.pdf"])
    assert corpus_glossary.scope_has_files(ids[:1], idx, CLOUD) is False


def test_unknown_or_out_of_range_ids_are_ignored():
    ids, idx = _idx(["excamera-nsdi17.pdf"])
    idx["pos"]["ghost"] = 99                   # θέση πέρα από τα metas
    assert corpus_glossary.scope_has_files(["missing", "ghost"], idx, CLOUD) is False
    assert corpus_glossary.scope_has_files([], idx, CLOUD) is False
    assert corpus_glossary.scope_has_files(ids, {}, CLOUD) is False


def test_descriptor_lists_the_files_it_describes():
    with open(DESCRIPTOR, encoding="utf-8") as f:
        d = json.load(f)
    assert d["domain"]
    files = d.get("files")
    assert files and all(isinstance(x, str) and x.endswith(".pdf") for x in files)
    assert len(set(files)) == len(files)
