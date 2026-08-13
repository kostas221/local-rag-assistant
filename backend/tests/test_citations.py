"""Inline citations: ο κανόνας 6 και οι ετικέτες S1..SN.

ΤΙ ΠΡΟΣΤΑΤΕΥΕΙ — τρία πράγματα που καμία μετρική RAG δεν πιάνει:

  1. ΤΗ ΘΕΣΗ ΤΟΥ ΚΑΝΟΝΑ. Ο κανόνας μετρήθηκε ως κανόνας 6, ΜΕΤΑ τον κανόνα
     μη-ψευδαίσθησης. Η θέση μέσα στο prompt έχει μετρηθεί ότι μετράει
     (corrective_v2: ίδιο περιεχόμενο, άλλη θέση, χειρότερο αποτέλεσμα σε κάθε
     κατώφλι). Αν κάποιος τον μετακινήσει, τα καταγεγραμμένα νούμερα παύουν να
     ισχύουν ΣΙΩΠΗΛΑ.

  2. ΤΗΝ ΕΥΘΥΓΡΑΜΜΙΣΗ ΕΤΙΚΕΤΑΣ - ΠΗΓΗΣ. Οι ετικέτες αριθμούνται ΠΡΙΝ το dedup,
     το UI εμφανίζει τις πηγές ΜΕΤΑ. Ένα enumerate() στο frontend θα έδειχνε
     [2] για μια απάντηση που λέει [S3] — δηλαδή θα έστελνε τον χρήστη σε
     ΛΑΘΟΣ σελίδα ενώ η παραπομπή είναι σωστή. Η ετικέτα ταξιδεύει ΜΕ το
     αντικείμενο ακριβώς γι' αυτό.

  3. ΤΟ CONTROL ΤΟΥ PROBE. Το probe_inline_citations.py --production παίρνει το
     σκέλος αναφοράς θέτοντας CITATION_RULE = "". Αν αυτό πάψει να σβήνει τον
     κανόνα, το probe θα μετράει ΤΟ ΙΔΙΟ ΠΡΑΓΜΑ ΔΥΟ ΦΟΡΕΣ και θα βγάζει
     «καμία διαφορά» — το χειρότερο είδος σφάλματος, γιατί μοιάζει με εύρημα.

ΜΗΔΕΝ δίκτυο: η γέννηση είναι stubbed και η ανάκτηση παρακάμπτεται με
`precomputed`, οπότε τα tests βλέπουν ΑΚΡΙΒΩΣ το prompt που θα έφευγε.
"""
import asyncio

import ai_core
import gemini_rest

PAGES = [
    ("Serverless computing removes server management.",
     {"file_name": "paper-a.pdf", "page": 3}),
    ("Cold starts dominate the tail latency.",
     {"file_name": "paper-b.pdf", "page": 11}),
]


def _drive(monkeypatch, precomputed=PAGES, question="what is serverless?"):
    """Τρέχει το ask_ai ως το τέλος και γυρνά (prompt, sources)."""
    seen = {}

    async def fake_stream(prompt, **kwargs):
        seen["prompt"] = prompt
        yield "text", "stub answer"

    monkeypatch.setattr(gemini_rest, "stream_generate", fake_stream)
    monkeypatch.setattr(ai_core, "USE_REST_GENERATION", True)

    async def run():
        out = []
        async for ev in ai_core.ask_ai(question, None, precomputed=precomputed):
            out.append(ev)
        return out

    events = asyncio.run(run())
    sources = next(e["data"] for e in events if e["type"] == "sources")
    return seen["prompt"], sources


def test_source_header_carries_a_label():
    assert ai_core.SOURCE_HEADER.format(i=1, file="a.pdf", page="7") == \
        "[S1: a.pdf, Page: 7]"


def test_citation_rule_asks_for_the_label_and_nothing_else():
    rule = ai_core.CITATION_RULE
    # Το παράδειγμα ΕΙΝΑΙ η προδιαγραφή: ό,τι δεν ζητάμε δεν μπορεί να βγει
    # λάθος. Ο αριθμός σελίδας βγήκε 10 φορές λάθος όσο τον ζητούσαμε.
    assert "[S3]" in rule
    assert "no page number" in rule
    assert "no file name" in rule
    assert "bibliography" in rule


def test_rule_six_comes_after_the_no_hallucination_rule(monkeypatch):
    prompt, _ = _drive(monkeypatch)
    assert "6. CITATIONS:" in prompt
    assert prompt.index("3. NO HALLUCINATIONS") < prompt.index("6. CITATIONS:")
    assert prompt.index("5. COMPLETENESS") < prompt.index("6. CITATIONS:")


def test_context_passages_are_labelled_in_order(monkeypatch):
    prompt, _ = _drive(monkeypatch)
    assert "[S1: paper-a.pdf, Page: 3]" in prompt
    assert "[S2: paper-b.pdf, Page: 11]" in prompt


def test_sources_carry_the_same_labels_as_the_context(monkeypatch):
    prompt, sources = _drive(monkeypatch)
    assert [s["label"] for s in sources] == ["S1", "S2"]
    for s in sources:
        assert f"[{s['label']}: {s['file']}, Page: {s['page']}]" in prompt


def test_labels_do_not_shift_when_dedup_drops_a_page(monkeypatch):
    """Η ΠΡΑΓΜΑΤΙΚΗ παγίδα: δύο περάσματα της ίδιας σελίδας.

    Το context παίρνει S1/S2/S3, το dedup αφήνει δύο πηγές. Η δεύτερη είναι η
    S3 — αν το UI την αρίθμαγε μόνο του θα την έλεγε [2].
    """
    dup = [
        ("first chunk", {"file_name": "paper-a.pdf", "page": 3}),
        ("second chunk", {"file_name": "paper-a.pdf", "page": 3}),
        ("other page", {"file_name": "paper-b.pdf", "page": 11}),
    ]
    prompt, sources = _drive(monkeypatch, precomputed=dup)
    assert [s["label"] for s in sources] == ["S1", "S3"]
    assert "[S3: paper-b.pdf, Page: 11]" in prompt


def test_empty_rule_restores_the_pre_citation_prompt(monkeypatch):
    """Το control του probe. Χωρίς αυτό η σύγκριση base/cite είναι άκυρη."""
    monkeypatch.setattr(ai_core, "CITATION_RULE", "")
    prompt, _ = _drive(monkeypatch)
    assert "CITATIONS" not in prompt
    assert "5. COMPLETENESS" in prompt
