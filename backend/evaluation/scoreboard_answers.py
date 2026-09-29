"""Φάση 0.3, 2ο μισό — ΑΠΑΝΤΗΣΕΙΣ και ΚΡΙΤΗΣ πάνω στον πίνακα αποτελεσμάτων.

ΑΦΟΡΜΗ (scoreboard, 28/9/2026): στις 29 νέες ερωτήσεις δύο εγγράφων η σελίδα-τεκμήριο του paper
που αναφέρεται 1ο στην ερώτηση έρχεται ~19/29, του 2ου ~11/29. Ο πίνακας μετράει ΣΕΛΙΔΕΣ.
Ανοιχτό: τι κάνει η ΑΠΑΝΤΗΣΗ όταν λείπει η σελίδα με το στοιχείο;
    (α) το λέει («τα έγγραφα δεν αναφέρουν…»)            -> σωστή συμπεριφορά
    (β) το προσπερνάει ή δίνει κάτι σχετικό αλλά όχι αυτό   -> υποβάθμιση
    (γ) το συμπληρώνει από γενική γνώση                    -> ψευδαίσθηση
    (δ) το βρίσκει σε ΑΛΛΗ σελίδα του ίδιου paper          -> τότε η μέτρηση σελίδων υποτιμά

ΤΙ ΚΑΝΕΙ: ίδια ανάκτηση με το scoreboard.py (απομονωμένο store, Gemini ανάκτησης παγωμένο ->
0 κλήσεις), ΕΛΕΓΧΟΣ ότι οι σελίδες είναι ΙΔΙΕΣ με τον πίνακα του ίδιου --label, μετά απάντηση με
τον ΠΡΑΓΜΑΤΙΚΟ ask_ai (ίδιο prompt, ίδιες ρυθμίσεις με την παραγωγή) και κριτής ΑΝΑ PAPER:
    correct · partial · declared_missing · omitted · unsupported
+ οι 4 βαθμοί 1-5 (ίδιοι ορισμοί με το run_eval). ΟΧΙ συγκρίσιμοι με παλιά judge runs: άλλο prompt.
Σειρά των papers = σειρά ΑΝΑΦΟΡΑΣ στην ερώτηση (ALIASES πάνω στο question_en), όχι η σειρά του
evidence_pages — διαφέρουν σε 2/29 (m071, m079).

ΠΑΓΩΜΑ: γέννηση ΚΑΙ κριτής με κλειδί ΟΛΟΚΛΗΡΟ το prompt, στο runs/scoreboard_answers.json.
Ξανατρέξιμο = 0 κλήσεις. Στη Φάση 1 όπου αλλάζουν οι σελίδες αλλάζει το prompt -> νέα γέννηση
και νέος κριτής ΜΟΝΟ εκεί. ΠΡΟΣΟΧΗ ΣΤΗΝ ΑΝΑΓΝΩΣΗ: η παγωμένη απάντηση είναι ΜΙΑ κλήρωση
(temperature 0.1, thinking 512) και ο κριτής δεν είναι ντετερμινιστικός (11/8: 2/9 γύρισαν κατά
μία μονάδα με ίδιο σύστημα). Μια αλλαγή ετικέτας σε ΜΙΑ ερώτηση δεν είναι εύρημα.

ΓΝΩΣΤΟ ΟΡΙΟ ΤΟΥ ΚΡΙΤΗ: η ετυμηγορία LLM είναι γλωσσοεξαρτώμενη (11/8: ελληνική ερώτηση πάνω σε
αγγλικό κείμενο -> «δεν απαντάει» ακόμα κι όταν απαντάει). Ο κριτής παίρνει την ΑΓΓΛΙΚΗ μορφή της
ερώτησης· η απάντηση και η απάντηση αναφοράς των ελληνικών μένουν ελληνικές (δεν υπάρχει αγγλική
αναφορά). Γι' αυτό τα αποτελέσματα τυπώνονται ΚΑΙ ανά γλώσσα — μια μεροληψία θα φαινόταν εκεί.

ΚΟΣΤΟΣ (1ο τρέξιμο): 29 γεννήσεις + 29 κριτές = 58 κλήσεις, ~0.35 $, ~6-8 λεπτά.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026) — 58 μισά, σελίδα-τεκμήριο ήρθε 30 / δεν ήρθε 28:
    ήρθε (30):      correct ≥ 24 · unsupported ≤ 1
    δεν ήρθε (28):  correct 5-11 · partial 6-12 · declared_missing 5-10 · omitted 1-5 ·
                    unsupported 0-3
    και τα δύο μισά correct: 9-15/29   (οι σελίδες λένε «και τα δύο papers» 7/29)
    faithfulness μέσος ≥ 4.6 · completeness μέσος 3.5-4.3
    Αν unsupported στα «δεν ήρθε» ≥ 4: το 2ο στρώμα (κανόνας 3 του prompt) ΔΕΝ κρατάει σε
    ερωτήσεις δύο μερών — η σύνθεση γεννάει ψευδαίσθηση, όπως η n024 των κοντινών ooc.

ΚΥΡΙΟ ΣΕΤ (--set main, golden_set_50, 50 ερωτήσεις). Κριτής: ο ΠΑΛΙΟΣ του eval_engine αυτούσιος
(ίδιο πρότυπο, ίδια κλήση SDK, temperature 0.0) -> οι βαθμοί συγκρίνονται με την κρίση της 11/8
(runs/judge_l12_full.csv: 4.94/4.92/5.00/4.90, 46/50 τέλειες, όχι τέλειες q027 q036 q046 q047).
CONTROL χωρίς Gemini: η κρίση της 11/8 μέσα από τη σύνοψη αυτού του script ξαναβγάζει ΑΚΡΙΒΩΣ
4.94/4.92/5.00/4.90 και 46/50. Οι κομμένες ερωτήσεις παίρνουν τη σταθερή άρνηση του ask_ai (μηδέν
γέννηση) και κρίνονται κανονικά, όπως στο run_eval.
ΚΟΣΤΟΣ: 44 γεννήσεις + 50 κριτές = 94 κλήσεις, ~0.55 $, ~10-12 λεπτά.
ΤΙ ΑΛΛΑΞΕ ΑΠΟ ΤΗΝ 11/8: μετάφραση + γλωσσάρι (16-18/8), παραπομπές [S3] (13/8), απομονωμένο store
(BM25 χωρίς ξένα αρχεία). Η q025 κόβεται πλέον από τον φύλακα (παλινδρόμηση μετάφρασης).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    τέλειες 5/5/5/5: 42-46/50   (η q025 σίγουρα όχι· από τις παλιές 4 ~2 ξανά, από θόρυβο κριτή)
    μέσοι: accuracy 4.82-4.94 · completeness 4.80-4.94 · relevance ≥ 4.90 · faithfulness 4.84-4.98
    q025: accuracy ≤ 2 (σταθερή άρνηση ενώ η απάντηση υπάρχει)
    out_of_corpus: 5/5 άρνηση, 5/5 τέλειες
    άλλαξαν βαθμούς έναντι 11/8: 4-10 ερωτήσεις
    ΣΥΝΑΓΕΡΜΟΣ: faithfulness < 5 σε ≥ 4 ερωτήσεις με απάντηση -> οι παραπομπές ή η νέα μετάφραση
    χάλασαν τη θεμελίωση (στις 11/8 ήταν 2: q046, q047).

ΚΟΝΤΙΝΕΣ ooc (--set near_ooc, golden_near_ooc, 42 ερωτήσεις). Κριτής: της Φάσης 0.1 αυτούσιος
(eval_near_ooc.JUDGE_PROMPT, ίδιες ρυθμίσεις — ο έλεγχος στο run() σκάει αν αλλάξουν). Επίπεδο 2
ΜΟΝΟ σε όσες περνούν τον φύλακα, όπως τότε. CONTROL χωρίς Gemini: το runs/near_ooc_eval.csv (25/9)
μέσα από τη σύνοψη αυτού του script ξαναβγάζει ΑΚΡΙΒΩΣ 2/38/2 · 38/1/1 · 10.8% · διαφωνίες
κριτή/μοτίβων n010 n048. Οι απαντήσεις είναι ΝΕΑ ΚΛΗΡΩΣΗ (η 25/9 δεν είχε πάγωμα) -> δείχνει και
πόσο σταθερό είναι το 38/1/1.
ΚΟΣΤΟΣ: 40 γεννήσεις + 40 κριτές = 80 κλήσεις, ~0.35 $, ~8 λεπτά.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    επίπεδο 1: 2 / 38 / 2 ΤΑΥΤΟΣΗΜΑ (ανάκτηση παγωμένη, έλεγχος σελίδων)
    επίπεδο 2: σωστό 36-40 · (β) 0-3 · (γ) 0-2
    n024 (γλώσσα -> autoscaling) ξανά (γ)· n048 (πάροχος -> ExCamera) ξανά (β)
    άλλαξε ετικέτα έναντι 25/9: 1-4 · κριτής ≠ μοτίβα άρνησης: 1-4
    ΣΥΝΑΓΕΡΜΟΣ: (γ) ≥ 3 -> το 1/42 της 25/9 ήταν τυχερή κλήρωση και το άνω όριο ανεβαίνει σε ~19%.

ΑΠΟΤΕΛΕΣΜΑ (28/9/2026, runs/scoreboard/answers_baseline_near_ooc.json):
    επίπεδο 1: 2 / 38 / 2 ✓ · επίπεδο 2: σωστό 38 · (β) 1 · (γ) 1 — ΙΔΙΟ ΣΥΝΟΛΟ με την 25/9 ✓
    n024 ξανά (γ) ✓ (ίδιο λάθος συμπέρασμα: «η γλώσσα δεν επηρεάζει το autoscaling») ·
    n048 ξανά (β) ✗ — τώρα ΣΩΣΤΟ· η (β) πήγε στη n038 (πολλαπλά cloud «το 2024»: η απάντηση άλλαξε
    σιωπηλά το ερώτημα σε «cloud» χωρίς τη φράση «δεν αναφέρεται το 2024 / multi-cloud» που είχε η
    1η κλήρωση — επιβεβαιωμένο με το μάτι) · άλλαξαν 2 ✓ · κριτής ≠ μοτίβα 1 (n003: το αγγλικό
    «it is not possible to determine» δεν είναι στα μοτίβα· ο κριτής σωστός) ✓ · συναγερμός όχι ✓.
    Έξι στις επτά. ΤΟ ΕΥΡΗΜΑ: η (γ) της n024 είναι ΣΤΑΘΕΡΟ σφάλμα (2/2 κληρώσεις), η (β) είναι
    ΚΛΗΡΩΣΗ (η φράση «λείπει» μπαίνει ή όχι σε οριακές: n048, n038 μία φορά η καθεμία). Στη Φάση 1
    μια (β) που μετακινείται ΔΕΝ είναι εύρημα· είναι αλλαγή συνόλου > ±1 ή ΝΕΑ (γ).

ΠΙΝΑΚΕΣ (--set tables, golden_tables, 16 ερωτήσεις, 69 τιμές). ΧΩΡΙΣ κριτή: οι τιμές ελέγχονται
με το eval_tables.check (ίδιες αποδεκτές παραλλαγές με την 13/8). Αποτυχία = «σύνθεση» αν ήρθε η
ΣΕΛΙΔΑ του πίνακα (doc+page του σετ), αλλιώς «ανάκτηση» — το eval_tables έκρινε με κάλυψη λέξεων.
CONTROL χωρίς Gemini: το runs/tables.csv (13/8) μέσα από τη σύνοψη -> 16/16, 69/69 ✓.
Προϋπόθεση: οι πίνακες στον πίνακα βάσης (scoreboard.py, 8ο σετ από 28/9).
ΚΟΣΤΟΣ: 16 γεννήσεις, ~0.09 $, ~3 λεπτά.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    πλήρως σωστές 14-16/16 · τιμές ≥ 66/69 · σύνθεση 0-2 · ανάκτηση 0-1
    άλλαξαν τιμές έναντι 13/8: 0-3   (τότε: ευρετήριο παραγωγής, χωρίς [S3], άλλη κλήρωση)
ΑΠΟΤΕΛΕΣΜΑ (28/9/2026): 16/16 σωστές ✓ · 69/69 τιμές ✓ · σύνθεση 0 ✓ · ανάκτηση 0 ✓ · άλλαξαν 0 ✓
    · σελίδες 16/16 ίδιες με τον πίνακα · tokens εξόδου διάμεσος 77. Πέντε στις πέντε.
    (Το 1ο τρέξιμο έγραψε «ήρθε η σελίδα του πίνακα 7/16» — ΣΦΑΛΜΑ ΣΕΤ, σελίδες από το 0· βλ.
    scoreboard.py. Δεν άλλαξε καμία ετυμηγορία: όλες ήταν «σωστή», που δεν εξαρτάται από τη σελίδα.)
    Οι πίνακες ΔΕΝ είναι πρόβλημα αυτού του συστήματος σε αυτό το σώμα — το q027 (13/8) ήταν
    εξαίρεση, όχι κανόνας. Μένει ως έλεγχος μη-υποβάθμισης για τη Φάση 1 (φθηνός: 16 κλήσεις).

ΑΛΛΑ ΠΕΔΙΑ (--set domains, golden_test_domains, 25 ερωτήσεις: 20 με απάντηση + 5 ooc, ΟΛΕΣ
ελληνικές). Στο ΔΙΚΟ ΤΟΥΣ store (scoreboard.open_domains_store, 100 chunks). Κριτής: του κύριου σετ
αυτούσιος (eval_engine) — οι απαντήσεις αναφοράς είναι ελληνικές, όπως στις ελληνικές του κύριου.
Οι κομμένες παίρνουν τη σταθερή άρνηση του ask_ai (μηδέν γέννηση). Προϋπόθεση: το 9ο σετ στον
πίνακα βάσης (scoreboard.py, 4ο τρέξιμο). Καμία προηγούμενη κρίση απαντήσεων σε αυτό το σετ.
ΚΟΣΤΟΣ: ~19-20 γεννήσεις + 25 κριτές ≈ 45 κλήσεις, ~0.25 $, ~5 λεπτά.
ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    σελίδες 25/25 ίδιες με τον πίνακα · τέλειες 5/5/5/5: 18-23/25
    μέσοι: accuracy ≥ 4.6 · faithfulness ≥ 4.8 · ooc 5/5 άρνηση, 5/5 τέλειες
    v3 (σιωπή ενώ η απάντηση υπάρχει): completeness ≤ 2 (όπως η q025: ο κριτής δίνει ακρίβεια 5 στη
    σωστή-κατά-το-context άρνηση, τιμωρεί μόνο την πληρότητα)
    ΣΥΝΑΓΕΡΜΟΣ: faithfulness < 5 σε ≥ 3 ερωτήσεις -> το prompt γέννησης (γραμμένο για cloud papers)
    δεν μεταφέρεται σε άλλο πεδίο.

    docker compose exec backend python evaluation/scoreboard_answers.py --set near_ooc
    docker compose exec backend python evaluation/scoreboard_answers.py --set tables
    docker compose exec backend python evaluation/scoreboard_answers.py --set domains
ΑΠΟΤΕΛΕΣΜΑ (28/9/2026, runs/scoreboard/answers_baseline_main.json, σελίδες 50/50 ίδιες):
    4.90 / 4.80 / 4.92 / 4.96 · τέλειες 47/50 ✗ (πρόβλεψη 42-46) · μέσοι μέσα στα εύρη ✓ ·
    ooc 5/5 ✓ · άλλαξαν 5 ✓ · συναγερμός όχι ✓ (στήριξη < 5 μόνο q046) · q025 ✗: ο κριτής
    έδωσε 5155 — ΑΚΡΙΒΕΙΑ 5 στη σταθερή άρνηση («σωστά είπες ότι δεν το βρίσκεις στο κείμενο που
    σου δόθηκε»). Αυτός ο κριτής ΔΕΝ τιμωρεί λάθος άρνηση στην ακρίβεια· μόνο στην πληρότητα.
    ΟΛΗ η διαφορά από την 11/8 είναι 5 ερωτήσεις (οι 45 ταυτόσημες): χειρότερα q025 (γνωστή
    παλινδρόμηση φύλακα), q047 4453 -> 1115, q046 4453 -> 4353 · καλύτερα q027, q036. Χωρίς
    q025/q047 οι άλλες 48 είναι ίσες ή λίγο καλύτερες σε ΚΑΙ ΤΟΥΣ 4 βαθμούς.
    q047 ΜΕ ΤΟ ΜΑΤΙ: ήρθαν ΚΑΙ οι δύο σελίδες-τεκμήρια (Baldini σ.7 1η, Berkeley19 σ.6 5η)· η
    απάντηση βγήκε δύο προτάσεις (121 tokens), σωστές και θεμελιωμένες, αλλά προσπέρασε το
    Berkeley view και τα ονόματα των εξαρτημάτων. Ο κριτής σωστός στην κατεύθυνση, υπερβολικός
    στο μέγεθος (ακρίβεια 1 σε σωστό περιεχόμενο). ΜΙΑ κλήρωση -> παρατήρηση, όχι εύρημα· ίδιο
    μοτίβο με τα multi_hop (χάνεται το 2ο μισό), εδώ όμως στη ΓΕΝΝΗΣΗ, όχι στην ανάκτηση.

    docker compose exec backend python evaluation/scoreboard_answers.py --set main --limit 3   # δοκιμή
    docker compose exec backend python evaluation/scoreboard_answers.py --set main
    docker compose exec backend python evaluation/scoreboard_answers.py --set mh_new      # 0 κλήσεις αν δεν άλλαξε τίποτα
    docker compose exec backend python evaluation/scoreboard_answers.py --set main --label X --compare evaluation/runs/scoreboard/answers_baseline_main.json
"""
import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as S

import gemini_rest

ANS_PATH = os.path.join(S.RUNS, "scoreboard_answers.json")
MH_PATH = os.path.join(E.HERE, "golden_multihop_v2.jsonl")
MAIN_PATH = os.path.join(E.HERE, "golden_set_50.jsonl")
HISTORY_MAIN = os.path.join(S.RUNS, "judge_l12_full.csv")   # τελευταίο πλήρες judge, 11/8/2026
LABELS = ("correct", "partial", "declared_missing", "omitted", "unsupported")
GR = {"correct": "σωστό", "partial": "μερικό", "declared_missing": "«λείπει» (το λέει)",
      "omitted": "το προσπερνάει", "unsupported": "χωρίς στήριξη"}
SCORES = ("accuracy", "completeness", "relevance", "faithfulness")
JUDGE_KW = {"temperature": 0.0, "max_output_tokens": 4096, "thinking_budget": 1024}

# Πώς ονομάζουν οι 29 ερωτήσεις κάθε paper (question_en) -> σειρά αναφοράς.
ALIASES = {
    "1702.04024.pdf": ("pywren",),
    "1706.03178.pdf": ("baldini",),
    "1812.03651.pdf": ("cidr", "hellerstein", "one step forward"),
    "1902.03383v1.pdf": ("view on serverless", "berkeley view (2019)", "jonas",
                         "cloud programming simplified"),
    "EECS-2009-28.pdf": ("above the clouds", "view of cloud computing"),
    "excamera-nsdi17.pdf": ("excamera",),
    "mapreduce-osdi04.pdf": ("mapreduce",),
}
NAMES = {
    "1702.04024.pdf": "PyWren (Jonas et al., 'Occupy the Cloud', 2017)",
    "1706.03178.pdf": "Baldini et al., 'Serverless Computing: Current Trends and Open Problems' (2017)",
    "1812.03651.pdf": "Hellerstein et al., 'Serverless Computing: One Step Forward, Two Steps Back' "
                      "(CIDR 2019)",
    "1902.03383v1.pdf": "Jonas et al., 'Cloud Programming Simplified: A Berkeley View on Serverless "
                        "Computing' (2019)",
    "EECS-2009-28.pdf": "Armbrust et al., 'Above the Clouds: A Berkeley View of Cloud Computing' (2009)",
    "excamera-nsdi17.pdf": "ExCamera (Fouladi et al., NSDI 2017)",
    "mapreduce-osdi04.pdf": "MapReduce (Dean & Ghemawat, OSDI 2004)",
}

MH_JUDGE = """You are grading one answer of a question-answering system that must answer ONLY \
from the SOURCE TEXT it was given (pages of research papers).

The QUESTION needs information from TWO papers:
  PAPER 1 = {name1} - file {doc1}
  PAPER 2 = {name2} - file {doc2}

QUESTION: {question}
REFERENCE ANSWER (written by reading both complete papers; may be in Greek): {reference}
Key terms expected in the part about PAPER 1: {kw1}
Key terms expected in the part about PAPER 2: {kw2}

SOURCE TEXT the system was given. Each page starts with a header [S<n>: <file>, Page: <page>]:
<<<
{context}
>>>

SYSTEM ANSWER (may be in Greek; a marker like [S3] cites the page whose header is S3):
<<<
{answer}
>>>

For EACH paper, label the part of the SYSTEM ANSWER about that paper with exactly one of:
- "correct": gives that paper's part of the reference answer, and the SOURCE TEXT supports it.
- "partial": gives only some of that part, or gives related information from that paper that \
the SOURCE TEXT supports but that misses the main point of the reference.
- "declared_missing": clearly says that the documents do not contain that part of the answer, \
and gives nothing useful for it.
- "omitted": does not address that paper's part and does not say that it is missing.
- "unsupported": states something about that paper's part that the SOURCE TEXT does not \
support, or that contradicts the reference - even if it happens to be true.
If the answer gives some useful information for a part AND says the rest is missing, the label \
is "partial".

Then score the WHOLE answer from 1 to 5:
- accuracy: factual correctness vs the reference answer.
- completeness: covers the key information of the reference.
- relevance: directly answers the question, no filler.
- faithfulness: EVERY claim in the answer is supported by the SOURCE TEXT.

Return ONLY JSON, no prose:
{{"paper_1": "<label>", "paper_2": "<label>",
 "evidence_1": "<ONE sentence copied VERBATIM from the SYSTEM ANSWER that decided the label of PAPER 1, or empty>",
 "evidence_2": "<the same for PAPER 2>",
 "accuracy": 0, "completeness": 0, "relevance": 0, "faithfulness": 0,
 "feedback": "<at most two sentences>"}}
"""

# ΚΥΡΙΟ ΣΕΤ: ο κριτής του eval_engine.evaluate_answer ΑΥΤΟΥΣΙΟΣ (ίδιο πρότυπο, ίδια κλήση μέσω
# SDK, response_mime_type json, temperature 0.0) -> οι βαθμοί συγκρίνονται με κάθε παλιό judge run
# (11/8: 4.94/4.92/5.00/4.90, 46/50 τέλειες). Ο έλεγχος στο run() σκάει αν ο eval_engine αλλάξει.
MAIN_JUDGE = """You are a strict expert evaluator for a RAG system. Only give 5/5 for perfect answers.

QUESTION: {question}

RETRIEVED CONTEXT (the ONLY source the system was allowed to use):
{context}

GENERATED ANSWER: {answer}

REFERENCE ANSWER (gold): {reference}

Evaluate (each 1-5):
1. accuracy: factual correctness vs the reference answer.
2. completeness: covers the key information of the reference.
3. relevance: directly answers the question, no filler.
4. faithfulness: EVERY claim in the generated answer is supported by the RETRIEVED CONTEXT. If the answer adds facts NOT in the context (hallucination), score this LOW even if they happen to be correct.

Return ONLY raw JSON (no markdown):
{{
    "feedback": "your detailed reasoning",
    "accuracy": 5,
    "completeness": 5,
    "relevance": 5,
    "faithfulness": 5
}}"""
_MAIN_PROBES = ("You are a strict expert evaluator for a RAG system.",
                "2. completeness: covers the key information of the reference.",
                "4. faithfulness: EVERY claim in the generated answer is supported",
                'context = "\\n---\\n".join(text for text, meta in retrieved)',
                "temperature=0.0")


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class AnswerCache:
    """Πάγωμα ΓΕΝΝΗΣΗΣ (stream_generate) και ΚΡΙΤΗ με κλειδί ολόκληρο το prompt.
    Χωριστό αρχείο από το πάγωμα της ανάκτησης: εκείνο δεν πρέπει να αλλάζει από εδώ."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.hits = self.misses = 0
        self.errors: list[str] = []
        self.last_key: str | None = None

    def install_stream(self, real_stream) -> None:
        async def frozen(prompt, **kw):
            key = _key(prompt)
            self.last_key = key
            if key in self.data:
                self.hits += 1
                yield "text", self.data[key]["out"]
                yield "usage", self.data[key].get("usage") or {}
                return
            text, usage = "", {}
            try:
                async for kind, d in real_stream(prompt, **kw):
                    if kind == "text":
                        text += d
                    else:
                        usage = d
                    yield kind, d
            except Exception as e:
                # Το ask_ai ΚΑΤΑΠΙΝΕΙ το σφάλμα και γράφει «Προσωρινό πρόβλημα…» στη θέση
                # της απάντησης -> χωρίς αυτό θα βαθμολογούσαμε το μήνυμα σφάλματος.
                self.errors.append(f"γέννηση: {type(e).__name__}: {e}"[:200])
                raise
            if not text.strip():
                self.errors.append("γέννηση: κενή απάντηση (μόνο thinking)")
                return
            self.misses += 1
            self.data[key] = {"kind": "answer", "tail": prompt[-200:], "out": text, "usage": usage}

        gemini_rest.stream_generate = frozen

    async def judge(self, prompt: str, call) -> str:
        """call: async (prompt) -> κείμενο. Κλειδί ΜΟΝΟ το prompt — κάθε κριτής έχει δικό του
        πρότυπο, άρα δύο κριτές δεν συγκρούονται."""
        key = _key(prompt)
        if key in self.data:
            self.hits += 1
            return self.data[key]["out"]
        text = await call(prompt)
        self.misses += 1
        self.data[key] = {"kind": "judge", "tail": prompt[-200:], "out": text}
        return text

    def tokens(self) -> tuple:
        usage = (self.data.get(self.last_key) or {}).get("usage") or {}
        return gemini_rest.usage_tokens(usage) if usage else (None, None, None)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


def mention_order(t: dict) -> list[str]:
    """Οι σελίδες-τεκμήρια με τη σειρά που αναφέρεται το paper τους στην ερώτηση."""
    q = t["question_en"].lower()
    pos = {}
    for ep in t["evidence_pages"]:
        doc = ep.rsplit(":", 1)[0]
        found = [q.find(a) for a in ALIASES[doc] if a in q]
        if not found:
            raise ValueError(f"{t['id']}: κανένα όνομα του {doc} στην ερώτηση")
        pos[ep] = min(found)
    if len(set(pos.values())) != len(pos):
        raise ValueError(f"{t['id']}: δύο papers στην ίδια θέση")
    return sorted(t["evidence_pages"], key=pos.get)


def half_signals(t: dict, ep: str, pages) -> dict:
    """Ντετερμινιστικά σήματα για ΕΝΑ μισό: ήρθε η σελίδα-τεκμήριο; κάποια σελίδα του paper;
    λέξη του paper μέσα σε σελίδα του ίδιου paper;"""
    doc = ep.rsplit(":", 1)[0]
    got = {f"{m.get('file_name')}:{m.get('page')}" for _x, m in pages}
    texts = [x.lower() for x, m in pages if m.get("file_name") == doc]
    kws = t["keywords_by_doc"][doc]
    return {"doc": doc, "ev": int(ep in got), "paper": int(bool(texts)),
            "kw": int(any(k.lower() in x for k in kws for x in texts))}


def context_of(ai_core, pages) -> str:
    return "\n".join(
        f"{ai_core.SOURCE_HEADER.format(i=i, file=m.get('file_name', '?'), page=m.get('page', '?'))}"
        f"\n{text}" for i, (text, m) in enumerate(pages, 1))


def parse_verdict(raw: str, answer: str) -> dict:
    obj = E.parse_json_object(raw)
    out = {}
    for h in (1, 2):
        lab = str(obj.get(f"paper_{h}", "")).strip().lower()
        out[f"label_{h}"] = lab if lab in LABELS else f"?{lab}"
        ev = str(obj.get(f"evidence_{h}") or "")
        out[f"evidence_{h}"] = ev
        out[f"evfound_{h}"] = int(len(ev.split()) >= 3 and E.squash(ev) in E.squash(answer))
    for s in SCORES:
        try:
            out[s] = int(obj.get(s))
        except (TypeError, ValueError):
            out[s] = None
    out["feedback"] = str(obj.get("feedback") or "")
    return out


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    halves = [(r, h) for r in ok for h in (1, 2)]
    sm = {"n": len(ok), "errors": len(rows) - len(ok)}
    for ev in (1, 0):
        sel = [r[f"label_{h}"] for r, h in halves if r[f"ev_{h}"] == ev]
        sm[f"ev{ev}"] = {lab: sel.count(lab) for lab in LABELS} | {"n": len(sel)}
    for h in (1, 2):
        sel = [r[f"label_{h}"] for r in ok]
        sm[f"order{h}"] = {lab: sel.count(lab) for lab in LABELS} | {
            "n": len(sel), "ev": sum(r[f"ev_{h}"] for r in ok)}
    for lang in ("el", "en"):
        sel = [r[f"label_{h}"] for r, h in halves if r["lang"] == lang]
        sm[f"lang_{lang}"] = {lab: sel.count(lab) for lab in LABELS} | {"n": len(sel)}
    both = [r for r in ok if r["label_1"] == r["label_2"] == "correct"]
    sm["both_correct"] = len(both)
    sm["both_correct_el"] = sum(r["lang"] == "el" for r in both)
    sm["both_correct_en"] = sum(r["lang"] == "en" for r in both)
    sm["n_el"] = sum(r["lang"] == "el" for r in ok)
    sm["n_en"] = sum(r["lang"] == "en" for r in ok)
    sm["both_pages"] = sum(r["both"] or 0 for r in ok)
    for s in SCORES:
        vals = [r[s] for r in ok if r[s] is not None]
        sm[s] = round(statistics.mean(vals), 2) if vals else None
    sm["perfect"] = sum(all(r[s] == 5 for s in SCORES) for r in ok)
    outs = [r["out_tokens"] for r in ok if r["out_tokens"]]
    sm["out_tokens_median"] = int(statistics.median(outs)) if outs else None
    return sm


def _cell(d: dict, lab: str) -> str:
    return f"{d[lab]}/{d['n']}" if d["n"] else "—"


def print_summary(sm: dict, prev: dict | None) -> None:
    print("\n" + "#" * 96)
    print(f"ΑΠΑΝΤΗΣΕΙΣ — {sm['n']} ερωτήσεις δύο εγγράφων ({2 * sm['n']} μισά)"
          + (f" · ΣΦΑΛΜΑΤΑ {sm['errors']}" if sm["errors"] else ""))
    print(f"{'ετικέτα κριτή ανά μισό':<26}{'σελίδα-τεκμήριο ήρθε':>22}{'δεν ήρθε':>12}"
          f"{'1ο paper':>12}{'2ο paper':>12}{'el':>9}{'en':>9}")
    for lab in LABELS:
        line = (f"{GR[lab]:<26}{_cell(sm['ev1'], lab):>22}{_cell(sm['ev0'], lab):>12}"
                f"{_cell(sm['order1'], lab):>12}{_cell(sm['order2'], lab):>12}"
                f"{_cell(sm['lang_el'], lab):>9}{_cell(sm['lang_en'], lab):>9}")
        if prev:
            line += (f"    πριν: {_cell(prev['ev1'], lab)} · {_cell(prev['ev0'], lab)}")
        print(line)
    print(f"{'(σελίδα-τεκμήριο ήρθε)':<26}{'':>22}{'':>12}"
          f"{sm['order1']['ev']:>9}/{sm['n']:<2}{sm['order2']['ev']:>9}/{sm['n']:<2}")
    print(f"\nκαι τα δύο μισά σωστά: {sm['both_correct']}/{sm['n']}  (el {sm['both_correct_el']}/"
          f"{sm['n_el']} · en {sm['both_correct_en']}/{sm['n_en']})   — οι σελίδες έλεγαν "
          f"«και τα δύο papers» {sm['both_pages']}/{sm['n']}"
          + (f"   πριν: {prev['both_correct']}/{prev['n']}" if prev else ""))
    print("βαθμοί (μέσος 1-5): " + " · ".join(f"{s} {sm[s]}" for s in SCORES)
          + f" · τέλειες 5/5/5/5: {sm['perfect']}/{sm['n']}"
          + ("\n              πριν: " + " · ".join(f"{s} {prev[s]}" for s in SCORES)
             + f" · τέλειες {prev['perfect']}/{prev['n']}" if prev else ""))
    print(f"tokens εξόδου (διάμεσος): {sm['out_tokens_median']}")
    print("#" * 96)


def print_attention(rows: list[dict]) -> None:
    """Ό,τι θέλει μάτι: κάθε «χωρίς στήριξη», κάθε άγνωστη ετικέτα, κάθε απόσπασμα που δεν βρέθηκε."""
    flagged = [(r, h) for r in rows if not r["error"] for h in (1, 2)
               if r[f"label_{h}"] == "unsupported" or r[f"label_{h}"].startswith("?")
               or (r[f"evidence_{h}"] and not r[f"evfound_{h}"])]
    if not flagged:
        return
    print("\nΓΙΑ ΕΛΕΓΧΟ ΜΕ ΤΟ ΜΑΤΙ:")
    for r, h in flagged:
        miss = "" if r[f"evfound_{h}"] or not r[f"evidence_{h}"] else "  [ΤΟ ΑΠΟΣΠΑΣΜΑ ΔΕΝ ΒΡΕΘΗΚΕ]"
        print(f"  {r['id']} {h}ο ({short_name(r[f'doc_{h}'])}, σελίδα-τεκμήριο "
              f"{'ήρθε' if r[f'ev_{h}'] else 'ΔΕΝ ήρθε'}): {r[f'label_{h}']}{miss}\n"
              f"      «{r[f'evidence_{h}'][:220]}»")


SHORT_DOC = {"1702.04024.pdf": "PyWren", "1706.03178.pdf": "Baldini", "1812.03651.pdf": "CIDR",
             "1902.03383v1.pdf": "Berkeley19", "EECS-2009-28.pdf": "Clouds09",
             "excamera-nsdi17.pdf": "ExCamera", "mapreduce-osdi04.pdf": "MapReduce"}


def short_name(doc: str) -> str:
    return SHORT_DOC[doc]


def print_flips(rows: list[dict], prev_rows: list[dict]) -> None:
    prev = {r["id"]: r for r in prev_rows}
    flips = []
    for r in rows:
        p = prev.get(r["id"])
        if not p or r["error"] or p["error"]:
            continue
        for h in (1, 2):
            if r[f"label_{h}"] != p[f"label_{h}"]:
                flips.append(f"  {r['id']} {h}ο ({short_name(r[f'doc_{h}'])}): {p[f'label_{h}']} -> "
                             f"{r[f'label_{h}']}   σελίδα-τεκμήριο {p[f'ev_{h}']} -> {r[f'ev_{h}']}")
    print(f"\nΑΛΛΑΓΕΣ ΕΤΙΚΕΤΑΣ: {len(flips)}")
    print("\n".join(flips) if flips else "  καμία")


# ----------------------------------------------------------------- ΚΥΡΙΟ ΣΕΤ (golden_set_50)

def check_main_judge(src: str) -> list[str]:
    """Γραμμές του MAIN_JUDGE που ΔΕΝ υπάρχουν στον κώδικα του eval_engine (εκτός από όσες έχουν
    placeholder, που εκεί γράφονται {test.question} κ.λπ.) + οι έλεγχοι context/θερμοκρασίας."""
    lines = [ln.strip() for ln in MAIN_JUDGE.splitlines()
             if ln.strip() and not re.search(r"(?<!\{)\{[a-z_]+\}(?!\})", ln)]
    return [x for x in lines + list(_MAIN_PROBES) if x not in src]


def parse_scores(raw: str) -> dict:
    obj = E.parse_json_object(raw)
    out = {}
    for s in SCORES:
        try:
            out[s] = int(obj.get(s))
        except (TypeError, ValueError):
            out[s] = None
    out["feedback"] = str(obj.get("feedback") or "")
    return out


def _means(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for s in SCORES:
        vals = [r[s] for r in rows if r[s] is not None]
        out[s] = round(statistics.mean(vals), 2) if vals else None
    out["perfect"] = sum(all(r[s] == 5 for s in SCORES) for r in rows)
    return out


def summarize_main(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    sm = {"n": len(ok), "errors": len(rows) - len(ok), "all": _means(ok),
          "in_corpus": _means([r for r in ok if r["category"] != "out_of_corpus"]),
          "by_cat": {c: _means([r for r in ok if r["category"] == c])
                     for c in dict.fromkeys(r["category"] for r in ok)},
          "by_lang": {g: _means([r for r in ok if r["lang"] == g]) for g in ("el", "en")}}
    ooc = [r for r in ok if r["category"] == "out_of_corpus"]
    sm["ooc_refused"] = sum(r["refusal"] for r in ooc)
    sm["ooc_n"] = len(ooc)
    outs = [r["out_tokens"] for r in ok if r["out_tokens"]]
    sm["out_tokens_median"] = int(statistics.median(outs)) if outs else None
    return sm


def _mline(name: str, m: dict, prev: dict | None = None) -> str:
    s = (f"  {name:<22} n={m['n']:<3} " + " · ".join(f"{m[s]:.2f}" if m[s] is not None else "—"
                                                    for s in SCORES)
         + f"   τέλειες {m['perfect']}/{m['n']}")
    if prev:
        s += ("      πριν: " + " · ".join(f"{prev[s]:.2f}" if prev[s] is not None else "—"
                                          for s in SCORES) + f"  τέλειες {prev['perfect']}/{prev['n']}")
    return s


def print_summary_main(sm: dict, prev: dict | None, title: str = "ΚΥΡΙΟ ΣΕΤ") -> None:
    print("\n" + "#" * 96)
    print(f"{title} — {sm['n']} ερωτήσεις" + (f" · ΣΦΑΛΜΑΤΑ {sm['errors']}" if sm["errors"] else ""))
    print(f"  {'':<22} {'':<5} accuracy · completeness · relevance · faithfulness")
    print(_mline("όλες", sm["all"], prev and prev["all"]))
    print(_mline("με απάντηση (όχι ooc)", sm["in_corpus"], prev and prev["in_corpus"]))
    for c, m in sm["by_cat"].items():
        print(_mline(c, m, prev and prev["by_cat"].get(c)))
    for g, m in sm["by_lang"].items():
        print(_mline(f"γλώσσα {g}", m, prev and prev["by_lang"].get(g)))
    print(f"out_of_corpus: απάντηση-άρνηση {sm['ooc_refused']}/{sm['ooc_n']}   ·   "
          f"tokens εξόδου (διάμεσος) {sm['out_tokens_median']}")
    print("#" * 96)


def print_nonperfect(rows: list[dict]) -> None:
    bad = [r for r in rows if not r["error"] and not all(r[s] == 5 for s in SCORES)]
    print(f"\nΟΧΙ 5/5/5/5: {len(bad)}")
    for r in bad:
        print(f"  {r['id']} {r['category']:<14} {r['lang']} {r.get('outcome', ''):<14} "
              + "".join(str(r[s]) for s in SCORES)
              + f"   {r['question'][:70]}\n      κριτής: {r['feedback'][:260]}")


def _tuple(r: dict) -> str:
    return "".join("?" if r.get(s) in (None, "") else str(int(float(r[s]))) for s in SCORES)


def print_flips_main(rows: list[dict], prev_rows: list[dict], title: str) -> None:
    prev = {r["id"]: r for r in prev_rows}
    flips = [f"  {r['id']} {r['category']:<14} {_tuple(prev[r['id']])} -> {_tuple(r)}"
             for r in rows if not r["error"] and r["id"] in prev and _tuple(prev[r["id"]]) != _tuple(r)]
    print(f"\n{title}: άλλαξαν βαθμούς {len(flips)}/{len(rows)}")
    print("\n".join(flips) if flips else "  καμία")


def load_history_main(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------ ΚΟΝΤΙΝΕΣ ooc (golden_near_ooc, Φάση 0.1)

NEAR_LABELS = E.LABELS                       # correct · soft_leak · leak
NEAR_GR = {"correct": "σωστό (λέει ότι λείπει)", "soft_leak": "ήπια διαρροή (β)",
           "leak": "διαρροή (γ)", "cut": "κόπηκε (σιωπή)"}


def parse_near(raw: str, answer: str) -> dict:
    obj = E.parse_json_object(raw)
    lab = str(obj.get("label", "")).strip().lower()
    ev = str(obj.get("evidence") or "")
    return {"label": lab if lab in NEAR_LABELS else f"?{lab}", "evidence": ev,
            "evfound": int(len(ev.split()) >= 3 and E.squash(ev) in E.squash(answer))}


def summarize_near(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    passed = [r for r in ok if r["outcome"] != "cut"]
    sm = {"n": len(ok), "errors": len(rows) - len(ok),
          "outcome": {o: sum(r["outcome"] == o for r in ok)
                      for o in ("cut", "passed_gate", "passed_corrective")},
          "passed": len(passed),
          "labels": {lab: sum(r["label"] == lab for r in passed) for lab in NEAR_LABELS},
          "unknown": sum(r["label"].startswith("?") for r in passed)}
    for lab in ("leak", "soft_leak"):
        sm[f"ub_{lab}"] = round(100 * E.upper_bound(sm["labels"][lab], len(ok)), 1) if ok else None
    for key, field in (("by_type", "near_type"), ("by_lang", "lang")):
        sm[key] = {v: {"n": sum(r[field] == v for r in passed),
                       "correct": sum(r[field] == v and r["label"] == "correct" for r in passed)}
                   for v in dict.fromkeys(r[field] for r in passed)}
    sm["regex_disagree"] = [r["id"] for r in passed if r["refusal"] != (r["label"] == "correct")]
    outs = [r["out_tokens"] for r in passed if r["out_tokens"]]
    sm["out_tokens_median"] = int(statistics.median(outs)) if outs else None
    return sm


def print_summary_near(sm: dict, prev: dict | None) -> None:
    print("\n" + "#" * 96)
    print(f"ΚΟΝΤΙΝΕΣ ooc — {sm['n']} ερωτήσεις" + (f" · ΣΦΑΛΜΑΤΑ {sm['errors']}" if sm["errors"] else ""))
    o = sm["outcome"]
    print(f"επίπεδο 1 (φύλακας)   κόπηκαν {o['cut']} · πέρασαν από gate {o['passed_gate']} · "
          f"μέσω corrective {o['passed_corrective']}"
          + (f"      πριν: {prev['outcome']['cut']}/{prev['outcome']['passed_gate']}/"
             f"{prev['outcome']['passed_corrective']}" if prev else ""))
    lb = sm["labels"]
    print(f"επίπεδο 2 (απάντηση, {sm['passed']} που πέρασαν)   σωστό {lb['correct']} · ήπια διαρροή (β) "
          f"{lb['soft_leak']} · διαρροή (γ) {lb['leak']}"
          + (f" · ΑΓΝΩΣΤΗ ετικέτα {sm['unknown']}" if sm["unknown"] else "")
          + (f"      πριν: {prev['labels']['correct']}/{prev['labels']['soft_leak']}/"
             f"{prev['labels']['leak']}" if prev else ""))
    print(f"άνω όριο 95% σε {sm['n']}:  (γ) {lb['leak']}/{sm['n']} -> {sm['ub_leak']}% · "
          f"(β) {lb['soft_leak']}/{sm['n']} -> {sm['ub_soft_leak']}%")
    for key, name in (("by_type", "τύπος"), ("by_lang", "γλώσσα")):
        print(f"σωστό ανά {name}: " + " · ".join(f"{k} {v['correct']}/{v['n']}"
                                                  for k, v in sm[key].items()))
    print(f"κριτής ≠ μοτίβα άρνησης: {' '.join(sm['regex_disagree']) or 'καμία'}   ·   "
          f"tokens εξόδου (διάμεσος) {sm['out_tokens_median']}")
    print("#" * 96)


def print_attention_near(rows: list[dict]) -> None:
    flagged = [r for r in rows if not r["error"] and r["outcome"] != "cut"
               and (r["label"] != "correct" or r["refusal"] != 1 or not r["evfound"])]
    if not flagged:
        return
    print("\nΓΙΑ ΕΛΕΓΧΟ ΜΕ ΤΟ ΜΑΤΙ (όχι «σωστό», ή διαφωνία με τα μοτίβα άρνησης, ή απόσπασμα που δεν βρέθηκε):")
    for r in flagged:
        miss = "" if r["evfound"] else "  [ΤΟ ΑΠΟΣΠΑΣΜΑ ΔΕΝ ΒΡΕΘΗΚΕ]"
        print(f"  {r['id']} {r['near_type']:<7} {r['lang']} κριτής {r['label']:<10} μοτίβα άρνησης "
              f"{r['refusal']}{miss}\n      ερώτηση: {r['question'][:120]}\n"
              f"      «{r['evidence'][:220]}»")


def load_history_near(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [dict(h, label=h["judge"] or "cut") for h in csv.DictReader(f) if h["kind"] == "near"]


def print_flips_near(rows: list[dict], prev_rows: list[dict], title: str) -> None:
    prev = {r["id"]: r for r in prev_rows}
    flips = [f"  {r['id']} {r['near_type']:<7} {prev[r['id']]['label']} -> {r['label']}"
             for r in rows if not r["error"] and r["id"] in prev
             and prev[r["id"]]["label"] != r["label"]]
    print(f"\n{title}: άλλαξε ετικέτα {len(flips)}/{len(rows)}")
    print("\n".join(flips) if flips else "  καμία")


async def judge_near(cache, t, answer, real_once) -> dict:
    # Ο κριτής της Φάσης 0.1 (eval_near_ooc.judge) ΑΥΤΟΥΣΙΟΣ: ίδιο πρότυπο, ίδιες ρυθμίσεις.
    prompt = E.JUDGE_PROMPT.format(focus=t["focus"], note=t.get("review_note") or "",
                                   question=t["question"], question_en=t["question_en"],
                                   answer=answer)
    raw = await cache.judge(prompt, lambda p: real_once(p, model=E.MODEL, api_key=E.API_KEY,
                                                        **JUDGE_KW))
    return parse_near(raw, answer)


# ------------------------------------------------------------ ΠΙΝΑΚΕΣ (golden_tables, 16)

VERDICTS = ("σωστή", "σύνθεση", "ανάκτηση")   # σύνθεση = ήρθε η σελίδα, λείπει τιμή


def table_verdict(n_ok: int, n_all: int, target: int | None) -> str:
    return "σωστή" if n_ok == n_all else "σύνθεση" if target else "ανάκτηση"


def summarize_tables(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    sm = {"n": len(ok), "errors": len(rows) - len(ok),
          "verdicts": {v: sum(r["verdict"] == v for r in ok) for v in VERDICTS},
          "target": sum(r["target"] or 0 for r in ok),
          "values_ok": sum(r["values_ok"] for r in ok),
          "values_total": sum(r["values_total"] for r in ok),
          "by_shape": {s: {"n": sum(r["shape"] == s for r in ok),
                           "ok": sum(r["shape"] == s and r["verdict"] == "σωστή" for r in ok)}
                       for s in sorted({r["shape"] for r in ok})}}
    outs = [r["out_tokens"] for r in ok if r["out_tokens"]]
    sm["out_tokens_median"] = int(statistics.median(outs)) if outs else None
    return sm


def print_summary_tables(sm: dict, prev: dict | None) -> None:
    v = sm["verdicts"]
    print("\n" + "#" * 96)
    print(f"ΠΙΝΑΚΕΣ — {sm['n']} ερωτήσεις" + (f" · ΣΦΑΛΜΑΤΑ {sm['errors']}" if sm["errors"] else ""))
    print(f"πλήρως σωστές {v['σωστή']}/{sm['n']} · σφάλμα ΣΥΝΘΕΣΗΣ {v['σύνθεση']} (ήρθε η σελίδα, "
          f"λείπει τιμή) · σφάλμα ΑΝΑΚΤΗΣΗΣ {v['ανάκτηση']} (δεν ήρθε η σελίδα)"
          + (f"      πριν: {prev['verdicts']['σωστή']}/{prev['verdicts']['σύνθεση']}/"
             f"{prev['verdicts']['ανάκτηση']}" if prev else ""))
    print(f"μεμονωμένες τιμές {sm['values_ok']}/{sm['values_total']} "
          f"({100 * sm['values_ok'] / max(sm['values_total'], 1):.1f}%) · ήρθε η σελίδα του πίνακα "
          f"{sm['target']}/{sm['n']} · tokens εξόδου (διάμεσος) {sm['out_tokens_median']}"
          + (f"      πριν: τιμές {prev['values_ok']}/{prev['values_total']}" if prev else ""))
    print("σωστές ανά μορφή πίνακα: " + " · ".join(f"{s} {d['ok']}/{d['n']}"
                                                   for s, d in sm["by_shape"].items()))
    print("#" * 96)


def print_attention_tables(rows: list[dict]) -> None:
    bad = [r for r in rows if not r["error"] and r["verdict"] != "σωστή"]
    if not bad:
        return
    print("\nΛΕΙΠΟΥΝ ΤΙΜΕΣ (για έλεγχο με το μάτι — μπορεί να γράφτηκε αλλιώς, π.χ. «0,5» / «½»):")
    for r in bad:
        print(f"  {r['id']} {r['shape']:<18} {r['verdict']:<9} τιμές {r['values_ok']}/{r['values_total']}"
              f"  λείπει: {r['missing']}\n      ερώτηση: {r['question'][:110]}\n"
              f"      απάντηση: {' '.join(r['answer'].split())[:400]}")


def load_history_tables(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [dict(h, values_ok=int(h["values_ok"]), values_total=int(h["values_total"]))
                for h in csv.DictReader(f)]


def print_flips_tables(rows: list[dict], prev_rows: list[dict], title: str) -> None:
    prev = {r["id"]: r for r in prev_rows}
    flips = [f"  {r['id']} {r['shape']:<18} τιμές {prev[r['id']]['values_ok']}/"
             f"{prev[r['id']]['values_total']} -> {r['values_ok']}/{r['values_total']}"
             + (f"   λείπει τώρα: {r['missing']}" if r["missing"] else "")
             for r in rows if not r["error"] and r["id"] in prev
             and int(prev[r["id"]]["values_ok"]) != r["values_ok"]]
    print(f"\n{title}: άλλαξαν τιμές {len(flips)}/{len(rows)}")
    print("\n".join(flips) if flips else "  καμία")


SET_PATHS = {"mh_new": MH_PATH, "main": MAIN_PATH,
             "near_ooc": os.path.join(E.HERE, "golden_near_ooc.jsonl"),
             "tables": os.path.join(E.HERE, "golden_tables.jsonl"),
             "domains": os.path.join(E.HERE, "golden_test_domains.jsonl")}   # ΑΛΛΟ store
HISTORY_TABLES = os.path.join(S.RUNS, "tables.csv")          # eval_tables, 13/8/2026, ΠΑΡΑΓΩΓΗ
HISTORY_NEAR = os.path.join(S.RUNS, "near_ooc_eval.csv")      # Φάση 0.1, 25/9/2026


async def judge_mh(cache, t, row, pages, answer, ai_core, real_once) -> dict:
    d1, d2 = row["doc_1"], row["doc_2"]
    prompt = MH_JUDGE.format(
        name1=NAMES[d1], doc1=d1, name2=NAMES[d2], doc2=d2, question=t["question_en"],
        reference=t["reference_answer"], kw1=", ".join(t["keywords_by_doc"][d1]),
        kw2=", ".join(t["keywords_by_doc"][d2]), context=context_of(ai_core, pages),
        answer=answer)
    raw = await cache.judge(prompt, lambda p: real_once(p, model=E.MODEL, api_key=E.API_KEY,
                                                        **JUDGE_KW))
    return parse_verdict(raw, answer)


async def judge_main(cache, t, pages, answer, sdk_call) -> dict:
    # Ό,τι έκανε το eval_engine: ΑΡΧΙΚΗ ερώτηση, context = σκέτα κείμενα με «---», χωρίς κεφαλίδες.
    prompt = MAIN_JUDGE.format(question=t["question"],
                               context="\n---\n".join(text for text, _m in pages),
                               answer=answer, reference=t["reference_answer"])
    return parse_scores(await cache.judge(prompt, sdk_call))


def line_mh(t, row) -> str:
    return (f"  {t['id']} {t['lang']}  1ο {short_name(row['doc_1']):<11} ev {row['ev_1']} "
            f"{row['label_1'] or '-':<17} 2ο {short_name(row['doc_2']):<11} ev {row['ev_2']} "
            f"{row['label_2'] or '-':<17} ")


def line_main(t, row) -> str:
    return (f"  {t['id']} {row['category']:<14} {row['lang']} {row['outcome']:<14} "
            f"σελίδες {row['n_pages']}  ")


async def run(args) -> int:
    real_once = gemini_rest.generate_once       # ΠΡΙΝ από κατασκόπους/πάγωμα της ανάκτησης
    real_stream = gemini_rest.stream_generate
    mh, near, tables = args.set == "mh_new", args.set == "near_ooc", args.set == "tables"

    base_path = os.path.join(S.OUT_DIR, f"{args.label}.json")
    if not os.path.exists(base_path):
        print(f"!! Λείπει ο πίνακας {base_path} — τρέξε πρώτα: scoreboard.py --label {args.label}")
        return 1
    with open(base_path, encoding="utf-8") as f:
        base = {r["id"]: r for r in json.load(f)["rows"] if r["set"] == args.set}
    tests = E.load_jsonl(SET_PATHS[args.set])
    if args.limit:
        tests = tests[:args.limit]
    orders = {t["id"]: mention_order(t) for t in tests} if mh else {}  # σκάει ΠΡΙΝ τη φόρτωση

    ai_core, _near = E.open_store()
    # ΜΕΤΑ το open_store: ίδιο ai_core
    import eval_engine
    calculate_mrr, calculate_ndcg = eval_engine.calculate_mrr, eval_engine.calculate_ndcg
    if tables:
        # ΙΔΙΑ συνάρτηση ελέγχου τιμών με την 13/8
        import eval_tables as T

    sdk_call = None
    if near:
        src = inspect.getsource(E.judge)
        if not ("temperature=0.0, max_output_tokens=4096" in src and "thinking_budget=1024" in src
                and JUDGE_KW == {"temperature": 0.0, "max_output_tokens": 4096,
                                 "thinking_budget": 1024}):
            print("!! Ο κριτής του eval_near_ooc ΑΛΛΑΞΕ ρυθμίσεις — δεν συγκρίνεται με την 25/9")
            return 1
    if args.set in ("main", "domains"):        # ίδιος κριτής: eval_engine αυτούσιος
        missing = check_main_judge(inspect.getsource(eval_engine.evaluate_answer))
        if missing:
            print(f"!! Ο κριτής του eval_engine ΑΛΛΑΞΕ — δεν βρέθηκαν: {missing}")
            return 1

        async def sdk_call(prompt):
            resp = await eval_engine.judge_model.generate_content_async(
                prompt, generation_config=eval_engine.genai.GenerationConfig(
                    response_mime_type="application/json", temperature=0.0))
            return resp.text

    want = S.EXPECTED_CHUNKS
    if args.set == "domains":                   # ΜΕΤΑ το E.open_store: κατάσκοποι ΜΙΑ φορά
        want = S.DOMAINS_CHUNKS
        S.open_domains_store(ai_core)
    if ai_core.collection.count() != want:
        print(f"!! Το store δεν έχει {want} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()          # μεταφράσεις ΜΟΝΟ μέσα από το πάγωμα
    ai_core.ENABLE_CORRECTIVE = True            # όπως η παραγωγή
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)
    frozen.install()
    cache = AnswerCache(ANS_PATH)
    cache.install_stream(real_stream)

    rows: list[dict] = []
    print(f"\n{len(tests)} ερωτήσεις ({args.set}): ανάκτηση (παγωμένη) -> απάντηση -> κριτής\n")
    for t in tests:
        t0 = time.perf_counter()
        pages, b1, b2, rw, outcome = await E.retrieve(ai_core, t["question"])
        r0 = S.make_row(args.set, t, t["question"], (pages, b1, b2, rw, outcome, 0.0),
                        ai_core, calculate_mrr, calculate_ndcg)
        b = base.get(t["id"])
        row = {"id": t["id"], "lang": r0["lang"], "category": r0["category"],
               "question": t["question"], "outcome": outcome, "n_pages": len(pages),
               "same_pages": int(bool(b) and b["pages"] == r0["pages"]), "error": ""}
        if mh:
            row |= {"mh_type": t.get("mh_type", ""), "both": r0["both"]}
            for h, ep in enumerate(orders[t["id"]], 1):
                sig = half_signals(t, ep, pages)
                row |= {f"doc_{h}": sig["doc"], f"ev_{h}": sig["ev"], f"paper_{h}": sig["paper"],
                        f"kw_{h}": sig["kw"]}
        elif near:
            row |= {"near_type": t.get("near_type", ""), "best1": r0["best1"], "best2": r0["best2"]}
        elif tables:
            row |= {"shape": t.get("shape", ""), "doc": t["doc"], "page": t["page"],
                    "target": r0["target"], "cov": r0["cov"], "n_kw": r0["n_kw"]}
        else:
            row |= {"cov": r0["cov"], "n_kw": r0["n_kw"], "mrr": r0["mrr"]}
            if args.set == "domains":
                row |= {"paper": t.get("paper", ""), "target": r0["target"]}

        n_err = len(cache.errors)
        answer = await E.answer_of(ai_core, t["question"], pages)
        row["answer"] = answer
        row["refusal"] = int(not pages or bool(E.REFUSAL.search(answer)))
        row["prompt_tokens"], row["out_tokens"], row["thinking_tokens"] = (
            cache.tokens() if pages else (None, None, None))
        if len(cache.errors) > n_err:
            row["error"] = cache.errors[-1]
        elif near and not pages:
            row |= {"label": "cut", "evidence": "", "evfound": 1}   # επίπεδο 2 ΜΟΝΟ όσες πέρασαν
        elif tables:                            # ΧΩΡΙΣ κριτή: ακριβείς τιμές, ντετερμινιστικά
            found = T.check(answer, t["expected"])
            row |= {"values_ok": sum(found), "values_total": len(found),
                    "missing": "|".join((v[0] if isinstance(v, list) else str(v))
                                        for v, f in zip(t["expected"], found) if not f),
                    "verdict": table_verdict(sum(found), len(found), row["target"])}
        else:
            try:
                if mh:
                    row |= await judge_mh(cache, t, row, pages, answer, ai_core, real_once)
                elif near:
                    row |= await judge_near(cache, t, answer, real_once)
                else:
                    row |= await judge_main(cache, t, pages, answer, sdk_call)
            except Exception as e:
                row["error"] = f"κριτής: {type(e).__name__}: {e}"[:200]
                cache.errors.append(row["error"])
        if mh:
            for h in (1, 2):
                row.setdefault(f"label_{h}", "")
                row.setdefault(f"evidence_{h}", "")
                row.setdefault(f"evfound_{h}", 0)
        if near:
            row.setdefault("label", "")
            row.setdefault("evidence", "")
            row.setdefault("evfound", 0)
        elif tables:
            row.setdefault("values_ok", 0)
            row.setdefault("values_total", len(t["expected"]))
            row.setdefault("missing", "")
            row.setdefault("verdict", "")
        else:
            for s in SCORES:
                row.setdefault(s, None)
            row.setdefault("feedback", "")
        rows.append(row)
        cache.save()                            # μετά από ΚΑΘΕ ερώτηση: ένα 429 δεν χάνει τα πληρωμένα
        mark = "" if row["same_pages"] else "  !! ΑΛΛΕΣ ΣΕΛΙΔΕΣ από τον πίνακα"
        verdict = (row["label"] if near else
                   f"τιμές {row['values_ok']}/{row['values_total']} {row['verdict']}" if tables else
                   "/".join(str(row[s]) for s in SCORES))
        print((line_mh(t, row) if mh else line_main(t, row))
              + (f"ΣΦΑΛΜΑ {row['error']}" if row["error"] else verdict)
              + f"  {time.perf_counter() - t0:.1f}s{mark}", flush=True)

    summ = (summarize(rows) if mh else summarize_near(rows) if near else
            summarize_tables(rows) if tables else summarize_main(rows))
    prev = prev_rows = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            pj = json.load(f)
        prev, prev_rows = pj["summary"], pj["rows"]
    if mh:
        print_summary(summ, prev)
        print_attention(rows)
        if prev_rows is not None:
            print_flips(rows, prev_rows)
    elif tables:
        print_summary_tables(summ, prev)
        print_attention_tables(rows)
        if prev_rows is not None:
            print_flips_tables(rows, prev_rows, "ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ --compare")
        if os.path.exists(HISTORY_TABLES) and not args.limit:
            print_flips_tables(rows, load_history_tables(HISTORY_TABLES),
                               "ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ eval_tables (13/8, runs/tables.csv)")
            print("  ^ ΔΕΝ είναι ίδιο πείραμα: τότε ευρετήριο ΠΑΡΑΓΩΓΗΣ, χωρίς παραπομπές [S3], άλλη "
                  "κλήρωση γέννησης.")
    elif near:
        print_summary_near(summ, prev)
        print_attention_near(rows)
        if prev_rows is not None:
            print_flips_near(rows, prev_rows, "ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ --compare")
        if os.path.exists(HISTORY_NEAR) and not args.limit:
            hist = load_history_near(HISTORY_NEAR)
            print_flips_near(rows, hist, "ΣΕ ΣΧΕΣΗ ΜΕ ΤΗ ΦΑΣΗ 0.1 (25/9, runs/near_ooc_eval.csv)")
            print("  ^ ΑΛΛΗ ΚΛΗΡΩΣΗ απάντησης και κριτή· 3/42 είχαν άλλη μετάφραση. Τότε: σωστό 38 · "
                  "(β) 1 (n048) · (γ) 1 (n024).")
    else:
        print_summary_main(summ, prev, "ΑΛΛΑ ΠΕΔΙΑ" if args.set == "domains" else "ΚΥΡΙΟ ΣΕΤ")
        print_nonperfect(rows)
        if prev_rows is not None:
            print_flips_main(rows, prev_rows, "ΣΕ ΣΧΕΣΗ ΜΕ ΤΟ --compare")
        if args.set == "main" and os.path.exists(HISTORY_MAIN) and not args.limit:
            hist = load_history_main(HISTORY_MAIN)
            print_summary_main(summarize_main(
                [dict(h, error="", lang=S.lang_of(h, h["question"]),
                      refusal=int(bool(E.REFUSAL.search(h["generated_answer"]))), out_tokens=None,
                      **{s: int(float(h[s])) for s in SCORES}) for h in hist]), None)
            print("  ^ η κρίση της 11/8 (runs/judge_l12_full.csv) με την ΙΔΙΑ σύνοψη. ΔΕΝ είναι ίδιο\n"
                  "    πείραμα: από τότε άλλαξαν μετάφραση/γλωσσάρι, παραπομπές [S3], store· και ο κριτής\n"
                  "    γυρίζει ~2/9 κατά μία μονάδα με ίδιο σύστημα.")
            print_flips_main(rows, hist, "ΣΕ ΣΧΕΣΗ ΜΕ ΤΗΝ ΚΡΙΣΗ ΤΗΣ 11/8")
    diff = [r["id"] for r in rows if not r["same_pages"]]
    print(f"\nΈΛΕΓΧΟΣ: ίδιες σελίδες με τον πίνακα «{args.label}» {len(rows) - len(diff)}/{len(rows)}"
          + (f" — ΔΙΑΦΕΡΟΥΝ: {' '.join(diff)}" if diff else " ✓"))
    print(f"Gemini: ανάκτηση {frozen.misses} νέες / {frozen.hits} παγωμένες · απαντήσεις+κριτής "
          f"{cache.misses} νέες / {cache.hits} παγωμένες")
    if cache.errors or frozen.errors:
        print(f"!! ΣΦΑΛΜΑΤΑ: {cache.errors + frozen.errors}")

    if args.limit:
        print("(--limit: δεν γράφεται αρχείο αποτελεσμάτων· οι πληρωμένες κλήσεις ΚΡΑΤΙΟΥΝΤΑΙ)")
        return 1 if (cache.errors or frozen.errors) else 0
    frozen.save()
    out = os.path.join(S.OUT_DIR, f"answers_{args.label}_{args.set}")
    judge_info = ({"model": E.MODEL, **JUDGE_KW} if mh or near else
                  {"none": "ακριβείς τιμές, eval_tables.check"} if tables else
                  {"model": E.MODEL, "via": "eval_engine SDK", "temperature": 0.0})
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "set": args.set, "when": time.strftime("%Y-%m-%d %H:%M"),
                   "judge": judge_info, "summary": summ, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    with open(out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"Σώθηκε: {out}.json / .csv")
    return 1 if (cache.errors or frozen.errors or diff) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", choices=sorted(SET_PATHS), default="mh_new")
    p.add_argument("--label", default="baseline",
                   help="ο πίνακας runs/scoreboard/<label>.json που δίνει τις σελίδες-έλεγχο")
    p.add_argument("--compare", help="answers_<άλλο>_<σετ>.json για σύγκριση")
    p.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N (δοκιμή)")
    args = p.parse_args()
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    sys.exit(main())
