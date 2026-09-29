"""Φάση 0.2 (2ο μισό) — ερωτήσεις ΔΥΟ ΕΓΓΡΑΦΩΝ: το multi_hop από 11 σε ~30.

ΤΟ ΕΡΩΤΗΜΑ ΜΕ ΑΠΛΑ ΛΟΓΙΑ:
    Οι 11 του golden_multihop_new είναι γραμμένες με το χέρι και κάθε μία ζητάει ένα
    κομμάτι από ένα paper ΚΑΙ ένα από άλλο. Με n=11 κάθε ερώτηση μετράει 9 μονάδες και
    δεν στηρίζεται κανένα συμπέρασμα (το AGENTS.md το λέει: «ακόμα μικρό»). Εδώ
    γράφονται ~20 ακόμα με ΤΗΝ ΙΔΙΑ συνταγή που έφτιαξε τα άλλα δύο σετ της Φάσης 0.

ΠΩΣ:
    1. ΓΕΝΝΗΣΗ ανά ΖΕΥΓΑΡΙ papers (7 -> 21 ζευγάρια): το Gemini βλέπει ΟΛΟΚΛΗΡΟ το
       κείμενο και των δύο και γράφει έως --n ερωτήσεις που θέλουν ΚΑΙ τα δύο. Αν το
       ζευγάρι δεν έχει φυσική σύνδεση, επιτρέπεται να μη γράψει καμία — μια
       στριμωγμένη ερώτηση δεν είναι αυτό που ρωτάει χρήστης. Τρεις τύποι:
         compare         — η ίδια πτυχή (μηχανισμός, αριθμός, σχεδιαστική επιλογή) στα δύο
         claim_evidence  — ισχυρισμός στο ένα, τι δείχνει το άλλο υπέρ ή κατά
         concept_link    — έννοια του ενός, πώς τη χρησιμοποιεί/μετράει/επεκτείνει το άλλο
       Για ΚΑΘΕ paper: μία πρόταση ΑΥΤΟΥΣΙΑ από τη σελίδα + 1-2 λέξεις-κλειδιά μέσα της.
    2. ΜΗΧΑΝΙΚΟΣ ΕΛΕΓΧΟΣ (μηδέν κόστος, ΠΡΙΝ την επαλήθευση — δεν πληρώνουμε για σκουπίδια):
         • η πρόταση υπάρχει αυτούσια στο paper που δηλώνει (σύγκριση χωρίς στίξη/γραμμές)·
           ή, αν τη διακόπτει υποσημείωση/αλλαγή σελίδας, ≥ 60% της συνεχόμενα (locate_quote).
           ΤΙΜΗΜΑ, συνειδητό: μία αλλαγμένη λέξη σε μακριά πρόταση περνάει πλέον (το test
           «αλλοιωμένο απόσπασμα» γύρισε σε ok). Το απόσπασμα χρησιμεύει να ΒΡΕΘΕΙ Η ΣΕΛΙΔΑ·
           την ύπαρξη του τεκμηρίου την ελέγχουν οι λέξεις-κλειδιά πάνω στη σελίδα + ο ελεγκτής.
         • κάθε λέξη-κλειδί υπάρχει στη σελίδα ΟΠΩΣ ΤΗΝ ΨΑΧΝΕΙ Η ΑΞΙΟΛΟΓΗΣΗ — με τις
           αλλαγές γραμμής. Φράση που σπάει σε δύο γραμμές πετιέται: αυτό ακριβώς ήταν
           το σφάλμα μέτρησης του hard set («next decade» -> 0 σελίδες).
         • όχι λέξη-κλειδί που υπάρχει ήδη στην ερώτηση (μετράει την ερώτηση, όχι την ανάκτηση)
         • όχι ΑΔΥΝΑΜΗ λέξη: αν 8 τυχαίες σελίδες τη βρίσκουν με πιθανότητα > WEAK_P, δεν
           μετράει ανάκτηση αλλά συχνότητα λέξης («serverless»: 99.8% τυχαία)
         • μένει ≥1 λέξη-κλειδί ΑΝΑ paper -> αλλιώς η ερώτηση πετιέται.
    3. ΕΠΑΛΗΘΕΥΣΗ με ΟΛΟ το σώμα στο context (αγγλική εκδοχή ερώτησης ΚΑΙ απάντησης —
       οι έλεγχοι με LLM είναι γλωσσοεξαρτώμενοι, probe_grounding_verdict 11/8):
         single          — μπορεί ΕΝΑ μόνο paper του σώματος να την απαντήσει ολόκληρη;
                           (τότε δεν είναι ερώτηση δύο εγγράφων — πετιέται)
         correct         — η απάντηση αναφοράς είναι σωστή και στηρίζεται στα papers;
         self_contained  — κατανοητή ως ΠΡΩΤΟ μήνυμα συνομιλίας;
         no_hint         — δεν προδίδει την απάντησή της;
    4. ΑΝΘΡΩΠΙΝΟΣ ΕΛΕΓΧΟΣ (runs/multihop_review.json): ο άνθρωπος ΜΟΝΟ πετάει, για
       ΕΓΚΥΡΟΤΗΤΑ. ΤΟ ΣΥΣΤΗΜΑ ΔΕΝ ΤΡΕΧΕΙ ΠΡΙΝ ΠΑΓΩΣΕΙ ΤΟ ΣΕΤ — αλλιώς θα πετούσαμε ό,τι
       αποτυγχάνει (η survivorship bias του «61/61 τέλειο»).

ΤΙ ΚΕΡΔΙΖΕΙ ΤΟ ΝΕΟ ΣΕΤ ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ ΠΑΛΙΟ: κάθε ερώτηση κρατάει ΞΕΧΩΡΙΣΤΑ τις
λέξεις-κλειδιά ΑΝΑ paper (keywords_by_doc). Έτσι μετριέται ΑΥΤΟ που είναι δύσκολο
στο multi_hop — ήρθαν σελίδες ΚΑΙ από τα δύο; — και όχι μόνο συνολική κάλυψη, όπου
μισή ανάκτηση (μόνο το ένα paper) φαίνεται «66%».

ΣΤΑΘΕΡΑ id (m001, m002, …): νέος γύρος ή «drop» δεν αλλάζει τα id των υπολοίπων.
Οι 11 παλιές ΔΕΝ αγγίζονται (συγκρισιμότητα με το ιστορικό MRR 0.493 / judge 5.00)·
οι νέες πάνε στο golden_multihop_v2.jsonl. Μαζί: ~30.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026), --n 2:
    υποψήφιες 34-42 (μερικά ζευγάρια δίνουν λιγότερες)
    μηχανικός έλεγχος περνάνε 60-80% — οι περισσότερες απώλειες σε φράσεις-κλειδιά που
        σπάνε σε γραμμές και σε αποσπάσματα που δεν είναι ακριβώς αυτούσια
    η επαλήθευση κρατάει ~70% από αυτές — κύρια απώλεια «απαντιέται από ΕΝΑ paper»
        (το Berkeley view 2019 συνοψίζει ExCamera/PyWren/CIDR)
    μηχανή κρατά 16-24 · μετά τον άνθρωπο 13-19 -> σύνολο με τις 11: 24-30
    πάτωμα τύχης του νέου σετ < 20% (το golden_multihop_new: 22.6%)

ΔΟΚΙΜΗ 1 ΖΕΥΓΑΡΙΟΥ (28/9, PyWren × Baldini): 2 υποψήφιες, 0 κρατήθηκαν.
    ✓ τα αποσπάσματα βγήκαν ΑΥΤΟΥΣΙΑ και τα 4 — το μεγαλύτερο ρίσκο του μηχανικού ελέγχου.
    ✗ m001: η ερώτηση ΕΛΕΓΕ ήδη το γεγονός του PyWren («το PyWren αναφέρει ότι … χρόνος
      εκκίνησης …») -> ο ελεγκτής «προδίδει την απάντηση»· και η απάντηση ισχυριζόταν ότι
      ο Baldini «προτείνει τεχνικές» ενώ απλώς λέει ότι χρειάζονται -> «εν μέρει σωστή».
    ✗ m002: τα keywords της πλευράς PyWren ήταν ΜΕΣΑ στην ερώτηση (pywren, aws lambda) ή
      αδύναμα (python) -> καμία ισχυρή· και η ερώτηση ήταν τετριμμένη («ποιος αναφέρει Lambda»).
    Το Gemini έδωσε 3 keywords ενώ ζητήθηκαν 1-2 (ακίνδυνο: κρατάμε έως 2).
    ΔΙΟΡΘΩΣΗ στο GEN_PROMPT: (1) η ερώτηση ΖΗΤΑΕΙ και τα δύο γεγονότα, δεν δηλώνει κανένα·
    (2) ουσία (μηχανισμός/αριθμός/όριο), όχι «ποιο προϊόν αναφέρεται»· (3) η απάντηση μόνο
    με ό,τι λέγεται ΡΗΤΑ· (4) keywords 1-2, ΠΟΥΘΕΝΑ στην ερώτηση. Η πρόβλεψη ΔΕΝ αλλάζει.

ΔΟΚΙΜΗ 3 ΖΕΥΓΑΡΙΩΝ (28/9, με τις διορθώσεις): 4 υποψήφιες (το PyWren × Baldini έδωσε 0),
0 κρατήθηκαν — αλλά ο ΕΛΕΓΧΟΣ δούλεψε σωστά σε 2 και ΛΑΘΟΣ σε 2:
    ✓ m001 (claim_evidence): η ερώτηση ΕΛΕΓΕ ότι το PyWren θεωρεί το S3 επαρκές για shuffle —
      ψέμα, το PyWren χρησιμοποιεί Redis ακριβώς επειδή δεν είναι. Ο ελεγκτής: «εν μέρει».
    ✓ m003 (claim_evidence): απέδωσε στο PyWren κείμενο του Berkeley (SIMD, 1902 σ.20) ->
      και τα δύο αποσπάσματα από το ίδιο αρχείο -> κόπηκε μηχανικά.
    ✗ m002 «μέγιστος χρόνος εκτέλεσης στο PyWren (300 s) vs CIDR (15 λεπτά)» — ΚΑΛΗ, αλλά η
      πρόταση του PyWren σπάει από υποσημείωση + αλλαγή σελίδας -> «όχι αυτούσιο».
      Διόρθωση: locate_quote (πρόθεμα/κατάληξη ≥ 60%).
    ✗ m004 «απόδοση S3 σε PyWren vs Berkeley» — ΚΑΛΗ, αλλά αγγλική χωρίς reference_answer_en
      -> «κενό πεδίο». Διόρθωση: αντιγραφή από το reference_answer.
    ΜΟΤΙΒΟ: και τα 2 compare βγήκαν καθαρές ερωτήσεις· και τα 2 claim_evidence ΔΗΛΩΝΑΝ έναν
    ισχυρισμό που το paper δεν κάνει. Η περιγραφή του τύπου ζητούσε «a claim in one paper» και
    νίκησε τον κανόνα (1). Διόρθωση: η ερώτηση ονομάζει το ΘΕΜΑ, ποτέ τον ισχυρισμό.

ΑΠΟΤΕΛΕΣΜΑ ΓΥΡΟΥ 1 (28/9/2026) — ΤΡΕΙΣ ΣΤΙΣ ΕΞΙ ΠΡΟΒΛΕΨΕΙΣ:
    υποψήφιες 38 (19/21 ζευγάρια)          πρόβλεψη 34-42   ✓
    μηχανικός έλεγχος 20 (53%)             60-80%           ✗  (11 χωρίς ισχυρή λέξη · 7 όχι αυτούσιο)
    μηχανή κράτησε 19                      16-24            ✓
    μετά τον άνθρωπο 8                     13-19            ✗
    σύνολο με τις 11: 19                   24-30            ✗
    πάτωμα τύχης 16.7%                     < 20%            ✓
    Κρατήθηκαν: compare 7 · claim_evidence 1 · concept_link 0 · el 1 / en 7 · 8/21 ζευγάρια.
    ΤΟ ΕΥΡΗΜΑ — Ο ΕΛΕΓΚΤΗΣ ΜΕ ΟΛΟ ΤΟ ΣΩΜΑ ΣΦΡΑΓΙΖΕΙ: πέρασε 20/21 με την ίδια γενική σημείωση
    («requires information from two different papers»). Ο άνθρωπος πέταξε 11/19 (runs/
    multihop_review.json, τεκμήρια grep): 4 δηλώνουν το μισό της απάντησης · 2 ψευδής
    προϋπόθεση (m023, m038: το ExCamera δεν λέει «straggler» πουθενά) · 1 λάθος απόδοση (m014:
    το 0.1 s είναι του Berkeley) · 1 ζητάει κάτι ανύπαρκτο (m020) · 3 διπλότυπα των παλιών 11.
    Η ΙΔΙΑ ερώτηση (m001) κόπηκε στο 1ο smoke ως «προδίδει την απάντηση» και πέρασε εδώ ->
    ίδιο εύρημα με το near_ooc (δίδυμα με αντίθετη ετυμηγορία). Ακρίβεια ελεγκτή 8/19 = 42%.
    ΜΟΤΙΒΟ ΑΝΑ ΤΥΠΟ: compare 7/10 επιβιώνουν τον άνθρωπο · claim_evidence 1/8 · concept_link 0/2.
    Οι τύποι που ζητούν «ισχυρισμό» γεννούν ψευδείς ισχυρισμούς ΚΑΙ δηλώνουν τη μισή απάντηση,
    παρά τον κανόνα (1) και την αλλαγή περιγραφής μετά το 2ο smoke. Και επειδή ο 1ος τύπος
    ανά ζευγάρι ήταν συνήθως αυτός, και η 1η γλώσσα ελληνικά -> έπεσαν μαζί τα ελληνικά (1/8).
    ΔΙΟΡΘΩΣΗ ΚΑΝΟΝΑ ΣΤΟ ΔΡΟΜΟ: near_question («cold starts» ~ «cold start») -> άλλαξε 1/38 (m018).

ΓΥΡΟΣ 2 — ΜΟΝΟ compare, ΜΟΝΟ ελληνικά (--append --types compare --langs el), 28/9/2026.
    Γιατί: compare 7/10 επιβιώνουν τον άνθρωπο (claim_evidence 1/8), και το σετ έχει 3/19
    ελληνικές σε σύστημα όπου η τελευταία παλινδρόμηση ήρθε από τη ΜΕΤΑΦΡΑΣΗ.
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ: υποψήφιες 30-42 · μηχανικός 45-60% · μηχανή κρατά 14-22 ·
    μετά τον άνθρωπο 8-12 νέες (θα βγουν επαναλήψεις του γύρου 1 παρά τη λίστα αποφυγής —
    ίδιο φαινόμενο με το near_ooc) · σύνολο 27-31 · ελληνικές ≥ 9 στο νέο σετ.
ΑΠΟΤΕΛΕΣΜΑ ΓΥΡΟΥ 2: υποψήφιες 42 ✓ · μηχανικός 69% ✗ · μηχανή 29 ✗ · μετά τον άνθρωπο 21 ✗ ·
    σύνολο 40 ✗ · ελληνικές 22 ✓. Δύο στις έξι — υποτίμησα ξανά: το compare επιβιώνει ~70%
    και στους δύο γύρους (7/10 -> 21/29), η πρόβλεψη είχε κρατήσει το ~42% του γύρου 1.
    Ο ελεγκτής πέρασε 29/29 με την ίδια γενική σημείωση — καμία απόρριψη.
    Ο άνθρωπος πέταξε 8: 2 αυτολεξεί διπλότυπα του γύρου 1 (m052 = m012, m056 = m016 — η λίστα
    αποφυγής ΔΕΝ τα σταμάτησε, και το φίλτρο διπλοτύπων συγκρίνει ΑΚΡΙΒΕΣ κείμενο) · 4 ζητούν
    στοιχείο που δεν υπάρχει (όριο μνήμης Baldini, μονάδα χρέωσης και διάρκεια ζωής μηχανών
    MapReduce, καθυστέρηση RPC MapReduce) · 1 ψευδής προϋπόθεση με λάθος απόδοση (m076) ·
    1 λάθος απάντηση (m075: το Above09 ΑΠΟΡΡΙΠΤΕΙ το «CapEx σε OpEx» που του αποδίδεται).
    ΚΟΡΕΣΜΟΣ: οι 29 ερωτήσεις του νέου σετ πατάνε σε μόλις 29 διαφορετικές σελίδες (58 θέσεις).
    Μόνο 2 έχουν και τις δύο σελίδες δικές τους· 15 μοιράζονται και τις δύο με άλλες. PyWren σ.3
    ×6, CIDR σ.3 ×5. Ο generator γυρίζει στα ίδια «πρωτοσέλιδα» στοιχεία (όρια, τιμές, stragglers)
    -> οι αποτυχίες θα είναι ΣΥΣΧΕΤΙΣΜΕΝΕΣ και το πραγματικό n μικρότερο από 29. Επικάλυψη = note,
    όχι drop (κανόνας της m037). 3ος γύρος ΔΕΝ προτείνεται: θα έφερνε κι άλλες επαναλήψεις.

ΚΟΣΤΟΣ (σώμα 103k tokens, μετρημένο 28/9): 21 γεννήσεις × ~30k (δύο papers) ≈ 0.65M +
επαλήθευση ~4 παρτίδες × ~110k ≈ 0.45M -> ~1.1M tokens εισόδου. Χρόνος ~15 λεπτά.
Καμία επαφή με την παραγωγή ή τη ChromaDB (δεν φορτώνει μοντέλα).

    # δοκιμή ΑΚΡΟ-ΕΩΣ-ΑΚΡΟ σε 1 ζευγάρι (1 γέννηση + 1 επαλήθευση), γράφει *_smoke:
    docker compose exec backend python evaluation/build_multihop.py --pairs 1
    # πλήρες:
    docker compose exec backend python evaluation/build_multihop.py
    # ΝΕΟΣ γύρος (επαληθεύει ΜΟΝΟ τις νέες):
    docker compose exec backend python evaluation/build_multihop.py --append
    # ξανά επαλήθευση όσων πέρασαν τον μηχανικό έλεγχο, ΧΩΡΙΣ νέα γέννηση:
    docker compose exec backend python evaluation/build_multihop.py --reuse-candidates
    # ΜΗΔΕΝ κλήσεις: ο ανθρώπινος έλεγχος -> jsonl:
    docker compose exec backend python evaluation/build_multihop.py --apply-review
"""
import argparse
import asyncio
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_near_ooc as B

import gemini_rest

HERE = B.HERE
MULTIHOP_OLD = os.path.join(HERE, "golden_multihop_new.jsonl")
REVIEW_PATH = os.path.join(HERE, "runs", "multihop_review.json")
TYPES = ("compare", "claim_evidence", "concept_link")
LANGS = ("el", "en")
MAX_PAGES = 8        # όσες σελίδες φτάνουν στο Gemini (ai_core.MAX_PAGES) -> πάτωμα τύχης
WEAK_P = 0.5         # λέξη που τη βρίσκουν 8 τυχαίες σελίδες με p > 0.5 = αδύναμη (~df > 10)
KW_PER_DOC = 2

# Οι τίτλοι του PDF metadata είναι άχρηστοι σε δύο papers («arXiv:1902…», «This paper is
# included…»)· ο generator χρειάζεται να ξέρει ΤΙ είναι το καθένα για να τα συνδέσει.
TITLES = {
    "1702.04024.pdf": "Occupy the Cloud: Distributed Computing for the 99% (PyWren, 2017)",
    "1706.03178.pdf": "Serverless Computing: Current Trends and Open Problems (Baldini et al., 2017)",
    "1812.03651.pdf": "Serverless Computing: One Step Forward, Two Steps Back (Hellerstein et al., CIDR 2019)",
    "1902.03383v1.pdf": "Cloud Programming Simplified: A Berkeley View on Serverless Computing (2019)",
    "EECS-2009-28.pdf": "Above the Clouds: A Berkeley View of Cloud Computing (2009)",
    "excamera-nsdi17.pdf": "Encoding, Fast and Slow: Low-Latency Video Processing Using Thousands "
                           "of Tiny Threads (ExCamera, NSDI 2017)",
    "mapreduce-osdi04.pdf": "MapReduce: Simplified Data Processing on Large Clusters (OSDI 2004)",
}

GEN_PROMPT = """You are building a TEST SET of CROSS-DOCUMENT questions for a \
question-answering system over this collection of research papers:

{overview}

Below is the COMPLETE text of TWO of these papers, page by page, with [file p.N] markers.

=== PAPER A: {file_a} ===
{text_a}

=== PAPER B: {file_b} ===
{text_b}
=== END ===

Write UP TO {n} questions that can ONLY be answered by combining a specific fact from \
paper A with a specific fact from paper B. If these two papers have no natural \
connection, return fewer questions or an empty array [] - do NOT force a question.
{type_rule}

These cross-document questions ALREADY exist. Do NOT repeat or paraphrase them:
{avoid}

Rules:
- A real user's question: natural phrasing, 1-2 sentences, understandable as the FIRST \
message of a conversation. Naming the systems or papers ("ExCamera", "the MapReduce \
paper") is fine.
- The question must NOT state what EITHER paper says - it ASKS for both facts. \
BAD: "Paper X reports <fact>. How does paper Y describe it?" (half of the answer is \
already in the question). GOOD: "What does X report about <aspect>, and how does Y \
<relate to it>?"
- Ask about substance a researcher cares about (a mechanism, a number, a trade-off, a \
limitation), not about which product or platform a paper mentions.
- Language: {lang_rule}
- "question_en": the English version (identical to "question" when lang is "en").
- "reference_answer": 1-3 sentences in the language of the question, combining BOTH facts. \
Every statement must be EXPLICITLY stated in the papers: never attribute to a paper \
something it only hints at (if a paper says techniques are needed but names none, do \
not say it proposes techniques). "reference_answer_en": the same in English.
- "evidence": exactly one entry for EACH paper, with "file", "page" (the N of the \
[file p.N] marker), "quote": ONE sentence copied VERBATIM from that page that supports \
the answer (exact copy - no ellipsis, no paraphrase, no merging of sentences), and \
"keywords": ONE or TWO terms, never more (prefer SINGLE words or numbers), that appear \
verbatim in that quote, belong to the ANSWER and appear NOWHERE in the question. Prefer \
rare technical terms over common words such as "serverless", "cloud", "storage", "data".

Return ONLY a JSON array, no prose:
[{{"type": "...", "lang": "el|en", "question": "...", "question_en": "...", \
"reference_answer": "...", "reference_answer_en": "...", \
"evidence": [{{"file": "...", "page": 1, "quote": "...", "keywords": ["..."]}}, \
{{"file": "...", "page": 1, "quote": "...", "keywords": ["..."]}}]}}]
"""

VERIFY_PROMPT = """Below is the COMPLETE text of a collection of research papers, page by \
page. After it there is a list of questions, each with a proposed reference answer.

For EACH question judge:
- "single": could ONE single paper of this collection answer the WHOLE question correctly \
and completely on its own? "yes" or "no". If "yes", put that paper's file name in \
"single_file".
- "correct": is the reference answer correct and supported by the papers? "yes", \
"partly" or "no".
- "self_contained": would a reader understand the question as the FIRST message of a \
conversation, without any other context? "yes" or "no".
- "no_hint": does the question avoid giving away its own answer? "yes" or "no".
- "note": one short sentence explaining any "yes" in "single" or any "no"/"partly".

=== PAPERS ===
{corpus}
=== END OF PAPERS ===

QUESTIONS:
{questions}

Return ONLY a JSON array, one object per question, in the same order:
[{{"id": "...", "single": "yes|no", "single_file": "", "correct": "yes|partly|no", \
"self_contained": "yes|no", "no_hint": "yes|no", "note": ""}}]
"""


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_review() -> dict:
    if not os.path.exists(REVIEW_PATH):
        return {"drop": {}, "notes": {}}
    with open(REVIEW_PATH, encoding="utf-8") as f:
        r = json.load(f)
    return {"drop": r.get("drop", {}), "notes": r.get("notes", {})}


def p_random(df: int, n_pages: int) -> float:
    """Πιθανότητα 8 τυχαίες σελίδες να περιέχουν λέξη που υπάρχει σε df σελίδες
    (υπεργεωμετρική, ίδια λογική με random_coverage_baseline.py)."""
    if df <= 0:
        return 0.0
    return 1 - math.comb(n_pages - df, MAX_PAGES) / math.comb(n_pages, MAX_PAGES)


TYPE_DESC = {
    "compare": '- "compare": the same aspect (a mechanism, a number, a design choice) in the '
               'two papers.',
    "claim_evidence": '- "claim_evidence": what one paper argues or predicts about a topic, and '
                      "whether the other paper's findings support or contradict it. The question "
                      "names the TOPIC, never the argument itself - the argument is part of the "
                      "answer.",
    "concept_link": '- "concept_link": a concept introduced in one paper, and how the other paper '
                    "uses, measures or extends it.",
}


def type_rule(types: tuple[str, ...]) -> str:
    """Με όλους τους τύπους = ΤΟ ΙΔΙΟ κείμενο με τον γύρο 1. Με έναν τύπο (γύρος 2: μόνο
    compare, που επιβίωσε 7/10 τον άνθρωπο έναντι 1/8 του claim_evidence) κάθε ερώτηση του
    ζευγαριού πρέπει να συγκρίνει ΑΛΛΗ πτυχή — αλλιώς θα βγουν δύο παραλλαγές της ίδιας."""
    if len(types) == 1:
        return (f'Question type: every question is "{types[0]}"; the questions of this pair '
                f'must be about DIFFERENT aspects.\n' + TYPE_DESC[types[0]])
    return ("Question types (use a different type for each question of this pair):\n"
            + "\n".join(TYPE_DESC[t] for t in types))


def lang_rule(n: int, langs: tuple[str, ...] = LANGS) -> str:
    order = [langs[i % len(langs)] for i in range(n)]
    return ("write the questions in this order of languages: "
            + ", ".join(f'#{i + 1} "{g}"' for i, g in enumerate(order))
            + ' ("el" = Greek, "en" = English).')


async def generate(pairs, papers: dict, n: int, avoid: list[str], delay: float,
                   types: tuple[str, ...] = TYPES, langs: tuple[str, ...] = LANGS) -> list[dict]:
    overview = "\n".join(f"- {f}: {t}" for f, t in TITLES.items())
    avoid_txt = "\n".join(f"- {q}" for q in avoid) or "(none)"
    out = []
    for k, (fa, fb) in enumerate(pairs):
        def text(f):
            return "\n\n".join(f"[{f} p.{i + 1}]\n{t}" for i, t in enumerate(papers[f]["pages"]))
        prompt = GEN_PROMPT.format(overview=overview, file_a=fa, file_b=fb,
                                   text_a=text(fa), text_b=text(fb), n=n,
                                   avoid=avoid_txt, lang_rule=lang_rule(n, langs),
                                   type_rule=type_rule(types))
        try:
            raw = await gemini_rest.generate_once(prompt, model=B.MODEL, api_key=B.API_KEY,
                                                  temperature=0.7, max_output_tokens=8192,
                                                  thinking_budget=1024)
            items = B.parse_json_array(raw)
        except Exception as e:   # ένα χαλασμένο ζευγάρι δεν ρίχνει τα υπόλοιπα
            print(f"  !! {fa} × {fb}: {e}", flush=True)
            items = []
        print(f"  [{k + 1}/{len(pairs)}] {fa} × {fb}: {len(items)} υποψήφιες", flush=True)
        for it in items:
            it["pair"] = [fa, fb]
            out.append(it)
        if k + 1 < len(pairs):
            await asyncio.sleep(delay)
    return out


def locate_quote(quote: str, pages_sq: list[str]) -> tuple[list[int], str]:
    """Σελίδες (1-based) όπου βρίσκεται το απόσπασμα + πώς βρέθηκε.
    Ολόκληρο, αλλιώς ΠΡΟΘΕΜΑ ή ΚΑΤΑΛΗΞΗ ≥ 60% των λέξεων (και ≥ 8): μια πρόταση που
    τη διακόπτει υποσημείωση ή αλλαγή σελίδας στο PDF είναι αυθεντική αλλά όχι συνεχής
    (smoke 28/9, PyWren σ.3: «…these limits will | 2A wren is much smaller… | 3 | be
    increased…»). Οι λέξεις-κλειδιά ελέγχονται ΧΩΡΙΣΤΑ πάνω στη σελίδα, άρα η χαλάρωση
    αφορά μόνο το ΠΟΥ, όχι το ΑΝ υπάρχει το τεκμήριο."""
    toks = B.el_tokenize(quote)

    def find(ts: list[str]) -> list[int]:
        s = " " + " ".join(ts) + " "
        return [i + 1 for i, p in enumerate(pages_sq) if s in p]

    hits = find(toks)
    if hits:
        return hits, "ολόκληρο"
    need = max(8, math.ceil(0.6 * len(toks)))
    for n in range(len(toks) - 1, need - 1, -1):
        for part, ts in (("πρόθεμα", toks[:n]), ("κατάληξη", toks[-n:])):
            hits = find(ts)
            if hits:
                return hits, f"{part} {n}/{len(toks)}"
    return [], ""


def near_question(k: str, qtok: list[str]) -> bool:
    """Λέξη-κλειδί που είναι ΚΛΙΤΙΚΟΣ τύπος λέξης της ερώτησης: «cold starts» με ερώτηση
    «cold start problem» (m018, 28/9) — δεν είναι υποσυμβολοσειρά, άρα περνούσε, ενώ μετράει
    την ερώτηση και όχι την ανάκτηση. Κάθε λέξη της (≥ 4 γράμματα) έχει στην ερώτηση λέξη που
    είναι πρόθεμά της ή το αντίστροφο, με διαφορά ≤ 3 γράμματα (κατάληξη, όχι άλλη λέξη:
    «part» ≠ «partition»). Φράσεις με κοντές λέξεις («i/o bound», «21× slower») δεν πιάνονται."""
    kt = B.el_tokenize(k)
    return bool(kt) and all(
        len(a) >= 4 and any(len(b) >= 4 and abs(len(a) - len(b)) <= 3
                            and (a.startswith(b) or b.startswith(a)) for b in qtok)
        for a in kt)


def mechanical(c: dict, papers: dict, page_sq: dict, df_of) -> str:
    """'ok' ή ο λόγος απόρριψης. Γράφει c['sides'] και c['kw_dropped'] (διαφάνεια)."""
    if c.get("type") not in TYPES or c.get("lang") not in LANGS:
        return "κακή μορφή (τύπος/γλώσσα)"
    # Σε αγγλική ερώτηση το Gemini παραλείπει κάποτε το «_en» ως περιττό (smoke 28/9: m004
    # χωρίς reference_answer_en). Είναι το ίδιο κείμενο εξ ορισμού -> αντιγραφή, όχι απόρριψη.
    if c["lang"] == "en":
        for src, dst in (("question", "question_en"), ("reference_answer", "reference_answer_en")):
            if not str(c.get(dst) or "").strip():
                c[dst] = c.get(src)
    if not all(str(c.get(k) or "").strip() for k in
               ("question", "question_en", "reference_answer", "reference_answer_en")):
        return "κακή μορφή (κενό πεδίο)"
    ev = c.get("evidence") or []
    if len(ev) != 2 or sorted(str(e.get("file")) for e in ev) != sorted(c["pair"]):
        return "δεν έχει ΕΝΑ απόσπασμα ανά paper"
    qtext = f"{c['question']} {c['question_en']}".lower()
    qtok = B.el_tokenize(qtext)
    sides, dropped = [], []
    for e in ev:
        f, quote = e["file"], str(e.get("quote") or "")
        if len(B.el_tokenize(quote)) < 6:
            return f"{f}: πολύ σύντομο απόσπασμα"
        hits, how = locate_quote(quote, page_sq[f])
        if not hits:
            return f"{f}: το απόσπασμα ΔΕΝ βρέθηκε αυτούσιο"
        try:
            declared = int(e.get("page"))
        except (TypeError, ValueError):
            declared = None
        page = declared if declared in hits else hits[0]
        text = papers[f]["pages"][page - 1].lower()
        kws: list[str] = []
        for k in e.get("keywords") or []:
            k = str(k).strip().lower()
            if not k or k in kws:
                continue
            if k not in text:
                dropped.append(f"{k}: όχι στη σελίδα όπως την ψάχνει η αξιολόγηση")
            elif k in qtext or near_question(k, qtok):
                dropped.append(f"{k}: υπάρχει ήδη στην ερώτηση")
            elif df_of(k)[1] > WEAK_P:
                dropped.append(f"{k}: αδύναμη (df {df_of(k)[0]}, τύχη {df_of(k)[1]:.0%})")
            else:
                kws.append(k)
        c["kw_dropped"] = dropped
        if not kws:
            return f"{f}: καμία ισχυρή λέξη-κλειδί"
        sides.append({"file": f, "page": page, "quote": quote, "quote_match": how,
                      "keywords": kws[:KW_PER_DOC]})
    c["sides"] = sides
    return "ok"


async def verify(cands: list[dict], corpus: str, batch: int, delay: float) -> dict:
    verdicts = {}
    for k in range(0, len(cands), batch):
        chunk = cands[k:k + batch]
        qs = "\n\n".join(f"{c['cid']}: {c['question_en']}\n   reference answer: "
                         f"{c['reference_answer_en']}" for c in chunk)
        raw = await gemini_rest.generate_once(VERIFY_PROMPT.format(corpus=corpus, questions=qs),
                                              model=B.MODEL, api_key=B.API_KEY, temperature=0.0,
                                              max_output_tokens=16384, thinking_budget=2048)
        for v in B.parse_json_array(raw):
            verdicts[str(v.get("id", "")).strip()] = v
        print(f"  επαλήθευση {k + len(chunk)}/{len(cands)}", flush=True)
        if k + batch < len(cands):
            await asyncio.sleep(delay)
    return verdicts


def decide(c: dict, v: dict | None, seen: set) -> tuple[str, str]:
    """(απόφαση, λόγος). ΣΥΝΤΗΡΗΤΙΚΟ: στην αμφιβολία πετιέται — μια ερώτηση που
    απαντά ΕΝΑ paper θα μετρούσε ως multi_hop ενώ δεν είναι."""
    key = B.squash(c["question_en"])
    if key in seen:
        return "drop", "διπλότυπο"
    seen.add(key)
    if c["mech"] != "ok":
        return "drop", c["mech"]
    if v is None:
        return "drop", "λείπει ετυμηγορία"
    fails = []
    if str(v.get("single", "")).lower() != "no":
        fails.append(f"την απαντά ΕΝΑ paper ({v.get('single_file') or '?'})")
    if str(v.get("correct", "")).lower() != "yes":
        fails.append(f"απάντηση: {v.get('correct')}")
    if str(v.get("self_contained", "")).lower() != "yes":
        fails.append("όχι αυτοτελής")
    if str(v.get("no_hint", "")).lower() != "yes":
        fails.append("προδίδει την απάντηση")
    return ("drop", "; ".join(fails)) if fails else ("keep", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="έως N ερωτήσεις ανά ζευγάρι")
    ap.add_argument("--pairs", type=int, default=0, help="μόνο τα πρώτα N ζευγάρια (δοκιμή)")
    ap.add_argument("--batch", type=int, default=10, help="ερωτήσεις ανά κλήση επαλήθευσης")
    ap.add_argument("--delay", type=float, default=30.0, help="s ανάμεσα στις επαληθεύσεις")
    ap.add_argument("--gen-delay", type=float, default=5.0, help="s ανάμεσα στις γεννήσεις")
    ap.add_argument("--types", default=",".join(TYPES), help="τύποι ερωτήσεων, με κόμμα")
    ap.add_argument("--langs", default=",".join(LANGS), help="γλώσσες κατά σειρά, με κόμμα")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--reuse-candidates", action="store_true")
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--apply-review", action="store_true")
    args = ap.parse_args()
    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    langs = tuple(g.strip() for g in args.langs.split(",") if g.strip())
    if not types or not set(types) <= set(TYPES) or not langs or not set(langs) <= set(LANGS):
        print(f"!! --types από {TYPES}, --langs από {LANGS} — σταματάω")
        return 1

    if not B.API_KEY and not args.apply_review:
        print("!! Λείπει GEMINI_API_KEY στο περιβάλλον — σταματάω")
        return 1
    plist = B.load_papers()
    papers = {p["file"]: p for p in plist}
    if sorted(papers) != sorted(TITLES):
        print(f"!! Περίμενα τα 7 PDF {sorted(TITLES)}, βρήκα {sorted(papers)} — σταματάω")
        return 1

    suffix = "_smoke" if args.pairs else ""
    cand_path = os.path.join(HERE, "runs", f"multihop_candidates{suffix}.json")
    out_path = os.path.join(HERE, f"golden_multihop_v2{suffix}.jsonl")

    corpus = "\n\n".join(f"[{p['file']} p.{i + 1}]\n{t}"
                         for p in plist for i, t in enumerate(p["pages"]))
    page_sq = {f: [B.squash(t) for t in p["pages"]] for f, p in papers.items()}
    all_pages = [t.lower() for p in plist for t in p["pages"]]
    n_pages = len(all_pages)
    df_cache: dict = {}

    def df_of(k: str) -> tuple[int, float]:
        if k not in df_cache:
            df = sum(k in t for t in all_pages)
            df_cache[k] = (df, p_random(df, n_pages))
        return df_cache[k]

    print(f"Σώμα: {len(plist)} papers · {n_pages} σελίδες · {len(corpus):,} χαρ "
          f"≈ {len(corpus) / 4.53:,.0f} tokens ανά κλήση επαλήθευσης")

    old: list[dict] = []
    if args.reuse_candidates or args.append or args.apply_review:
        if not os.path.exists(cand_path):
            print(f"!! Δεν υπάρχει {cand_path} — τρέξε πρώτα χωρίς σημαία")
            return 1
        with open(cand_path, encoding="utf-8") as f:
            old = json.load(f)
        print(f"Φορτώθηκαν {len(old)} υποψήφιες από {cand_path}")

    if args.reuse_candidates or args.apply_review:
        cands = old
        new = old if args.reuse_candidates else []
    else:
        pairs = list(itertools.combinations(sorted(TITLES), 2))
        pairs = pairs[:args.pairs] if args.pairs else pairs
        avoid = [r["question"] for r in load_jsonl(MULTIHOP_OLD)] + [c["question_en"] for c in old]
        print(f"\nΓέννηση ({len(pairs)} ζευγάρια × έως {args.n}, τύποι {types}, γλώσσες {langs}, "
              f"λίστα αποφυγής {len(avoid)}):")
        raw = asyncio.run(generate(pairs, papers, args.n, avoid, args.gen_delay, types, langs))
        next_id = max((int(c["cid"][1:]) for c in old), default=0) + 1
        rnd = max((c["round"] for c in old), default=0) + 1
        new = []
        for i, it in enumerate(raw):
            it["cid"] = f"m{next_id + i:03d}"
            it["round"] = rnd
            new.append(it)
        cands = old + new

    # Μηχανικός έλεγχος: ντετερμινιστικός, ξανατρέχει ΠΑΝΤΑ (και στο --apply-review) ώστε
    # μια αλλαγή κανόνα εδώ να μη θέλει νέα γέννηση.
    for c in cands:
        c["mech"] = mechanical(c, papers, page_sq, df_of)
    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)

    to_verify = [c for c in new if c["mech"] == "ok"]
    verdicts: dict = {}
    if to_verify:
        print(f"\nΕπαλήθευση {len(to_verify)} (από {len(new)} νέες· οι υπόλοιπες κόπηκαν "
              f"μηχανικά) με ΟΛΟΚΛΗΡΟ το σώμα, παρτίδες των {args.batch}:")
        verdicts = asyncio.run(verify(to_verify, corpus, args.batch, args.delay))

    new_ids = {c["cid"] for c in new}
    seen: set = set()
    for c in cands:
        if c["cid"] in new_ids:
            v = verdicts.get(c["cid"])
            c["verdict"] = v or {}
            c["decision"], c["reason"] = decide(c, v, seen)
        else:
            seen.add(B.squash(c["question_en"]))
            if c["mech"] != "ok":     # κανόνας άλλαξε από τότε -> ισχύει και για τις παλιές
                c["decision"], c["reason"] = "drop", c["mech"]
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)

    review = load_review()
    unknown = sorted((set(review["drop"]) | set(review["notes"])) - {c["cid"] for c in cands})
    if unknown:
        print(f"!! Ο έλεγχος αναφέρει cid που δεν υπάρχουν: {unknown}")
    machine_kept = [c for c in cands if c.get("decision") == "keep"]
    kept = [c for c in machine_kept if c["cid"] not in review["drop"]]

    shown = [c for c in cands if c["cid"] in new_ids] or machine_kept
    print(f"\n{'cid':<6}{'τύπος':<16}{'γλ':<4} απόφαση / ερώτηση")
    for c in shown:
        if c.get("decision") != "keep":
            mark = f"πετιέται: {c.get('reason', c['mech'])}"
        elif c["cid"] in review["drop"]:
            mark = f"πετιέται (άνθρωπος): {review['drop'][c['cid']]}"
        else:
            mark = "ΚΡΑΤΙΕΤΑΙ"
        print(f"{c['cid']:<6}{c.get('type')!s:<16}{c.get('lang')!s:<4} {mark}\n"
              f"{'':<10}{c.get('question', '')[:110]}")
        if c.get("decision") == "keep" and c["cid"] not in review["drop"]:
            for s in c["sides"]:
                print(f"{'':<10}· {s['file']} σ.{s['page']} {s['keywords']}: «{s['quote'][:120]}»")
            note = (c.get("verdict") or {}).get("note")
            if note:
                print(f"{'':<10}  ελεγκτής: {note}")
        for d in c.get("kw_dropped") or []:
            print(f"{'':<10}  λέξη εκτός: {d}")

    floors = []
    with open(out_path, "w", encoding="utf-8") as f:
        for c in kept:
            kw_by_doc = {s["file"]: s["keywords"] for s in c["sides"]}
            kws = [k for s in c["sides"] for k in s["keywords"]]
            floors.append(sum(df_of(k)[1] for k in kws) / len(kws))
            row = {
                "id": c["cid"], "question": c["question"], "keywords": kws,
                "keywords_by_doc": kw_by_doc, "reference_answer": c["reference_answer"],
                "category": "multi_hop", "mh_type": c["type"], "lang": c["lang"],
                "question_en": c["question_en"], "docs": sorted(kw_by_doc),
                "evidence_pages": [f"{s['file']}:{s['page']}" for s in c["sides"]],
                "round": c["round"],
            }
            if c["cid"] in review["notes"]:
                row["review_note"] = review["notes"][c["cid"]]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "#" * 78)
    for r in sorted({c["round"] for c in cands}):
        rc = [c for c in cands if c["round"] == r]
        mech_ok = sum(c["mech"] == "ok" for c in rc)
        single = sum(str((c.get("verdict") or {}).get("single", "")).lower() == "yes" for c in rc)
        print(f"γύρος {r}: υποψήφιες {len(rc)} · μηχανικός έλεγχος {mech_ok} "
              f"({mech_ok / max(1, len(rc)):.0%}) · «την απαντά ΕΝΑ paper» {single} · "
              f"μηχανή κράτησε {sum(c.get('decision') == 'keep' for c in rc)} · "
              f"μετά τον άνθρωπο {sum(c['round'] == r for c in kept)}")
    mech_reasons = [c["mech"].split(": ", 1)[-1] for c in cands if c["mech"] != "ok"]
    if mech_reasons:
        print("  μηχανικές απορρίψεις: " + " · ".join(
            f"{m} {mech_reasons.count(m)}" for m in sorted(set(mech_reasons))))
    print(f"ΣΥΝΟΛΟ ΣΤΟ ΝΕΟ ΣΕΤ: {len(kept)}   (+ 11 του golden_multihop_new = {len(kept) + 11})")
    print("  ανά τύπο:   " + " · ".join(f"{t} {sum(c['type'] == t for c in kept)}" for t in TYPES))
    print("  ανά γλώσσα: " + " · ".join(f"{g} {sum(c['lang'] == g for c in kept)}" for g in LANGS))
    pair_count = {tuple(sorted(c["pair"])) for c in kept}
    print(f"  ζευγάρια papers που καλύπτονται: {len(pair_count)} από 21")
    if floors:
        print(f"  πάτωμα τύχης (μέση πιθανότητα 8 τυχαίες σελίδες να βρουν λέξη): "
              f"{sum(floors) / len(floors):.1%}   (golden_multihop_new: 22.6%)")
    print("ΠΡΟΒΛΕΨΗ γύρου 1 (γραμμένη πριν): υποψήφιες 34-42 · μηχανικός 60-80% · μηχανή κρατά "
          "16-24 · μετά τον άνθρωπο 13-19 · πάτωμα < 20%.")
    print("ΠΡΟΒΛΕΨΗ γύρου 2 (compare, el): υποψήφιες 30-42 · μηχανικός 45-60% · μηχανή κρατά 14-22 · "
          "μετά τον άνθρωπο 8-12 νέες · σύνολο 27-31 · ελληνικές ≥ 9 στο νέο σετ.")
    print(f"\nΓράφτηκε: {out_path}\nΥποψήφιες + ετυμηγορίες: {cand_path}")
    print("ΕΠΟΜΕΝΟ: διάβασε τις ΚΡΑΤΗΜΕΝΕΣ· όποια κρίνεις άκυρη (λάθος, διφορούμενη, διπλότυπη),\n"
          f"πρόσθεσέ τη στο \"drop\" του {REVIEW_PATH} και τρέξε --apply-review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
