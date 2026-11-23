"""Κριτής επάρκειας της δεύτερης αναζήτησης (sufficiency.py): το ντετερμινιστικό μέρος. Χωρίς μοντέλο
-> fast CI. Κλειδώνουμε: (1) το prompt είναι αυτό του Αυγούστου, (2) οι σελίδες έχουν την ίδια
επικεφαλίδα με την απάντηση, (3) μόνο ένα ρητό ΟΧΙ κόβει — σφάλμα ή ασάφεια αφήνουν τη σημερινή
συμπεριφορά."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sufficiency as SF

HERE = os.path.dirname(os.path.abspath(__file__))


def test_prompt_is_the_august_one_verbatim():
    with open(os.path.join(HERE, "..", "evaluation", "probe_corrective_verdict.py"), encoding="utf-8") as f:
        august = f.read()
    assert SF.VERDICT_PROMPT in august


def test_pages_carry_the_answer_headers_and_the_original_question():
    pages = [("Lambda is billed per 100 ms.", {"file_name": "a.pdf", "page": 3}),
             ("S3 costs $0.15 per GB-month.", {"file_name": "b.pdf", "page": 11})]
    p = SF.build_prompt("What does GDPR require from cloud providers?", pages)
    assert "[S1: a.pdf, Page: 3]\nLambda is billed" in p
    assert "[S2: b.pdf, Page: 11]\nS3 costs" in p
    assert p.rstrip().endswith("QUESTION: What does GDPR require from cloud providers?\n\nVERDICT:")


def test_parse_accepts_common_formats_and_rejects_hedges():
    for raw, want in [("YES", "YES"), ("No.", "NO"), ("**NO**", "NO"), ("VERDICT: yes", "YES"),
                      ("  no\n", "NO"), ("NOT SURE", None), ("", None), (None, None),
                      ("Maybe", None), ("NOPE", None)]:
        assert SF.parse_verdict(raw) == want, raw


def test_only_an_explicit_no_vetoes():
    assert SF.vetoes("NO") is True
    assert SF.vetoes("YES") is False
    assert SF.vetoes(None) is False          # σφάλμα/ασάφεια -> σημερινή συμπεριφορά
