"""ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ — Φάση 0.3: ΟΛΑ τα σετ, ΜΙΑ εντολή, απομονωμένο store.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (28/9/2026):
    Κάθε σετ είχε δικό του script με δικό του τρόπο ανάκτησης, και το run_eval.py ψάχνει
    στο ευρετήριο της ΠΑΡΑΓΩΓΗΣ (ξένα αρχεία αλλάζουν τη BM25). Καμία αλλαγή δεν φαινόταν
    «πριν/μετά» σε όλα μαζί — και η παλινδρόμηση της μετάφρασης (16-18/8) ελέγχθηκε ΜΟΝΟ
    με coverage· ο φύλακας δεν ξαναμετρήθηκε για 6 εβδομάδες.
    Εδώ: ΕΝΑ store (τα 7 cloud PDF), ΕΝΑ πέρασμα ανά ερώτηση όπως η παραγωγή (με corrective),
    ΟΛΕΣ οι μετρικές σε έναν πίνακα, και --compare με προηγούμενο τρέξιμο: ποιες ερωτήσεις
    ΓΥΡΙΣΑΝ. Σε n=16-59 οι μέσοι όροι κρύβουν θόρυβο· η λίστα αλλαγών ανά ερώτηση όχι.

ΤΙ ΜΕΤΡΑΕΙ (επίπεδο 1 — ανάκτηση· κόστος μόνο μεταφράσεις/αναδιατυπώσεις, παγωμένες):
    κύριο         45 με απάντηση: MRR · nDCG@3 · κάλυψη λέξεων (ΙΔΙΟΙ τύποι με το eval_engine)
    φύλακας       56 με απάντηση (κύριο + multihop_new) + 5 ooc, στο 1ο πέρασμα
                  (ΙΔΙΟ σύνολο με το measure_gate_margin): σωστά, κενό, ποιες κόβονται άδικα
    multi_hop     11 + 29: κάλυψη· στις 29 και «ήρθαν σελίδες ΚΑΙ από τα δύο papers»
                  (λέξη του paper Χ μέσα σε σελίδα του paper Χ — όχι απλώς κάπου)
    δύσκολο       16 + 59: με υλικό / χωρίς υλικό / σιωπή (όπως η παραγωγή, με corrective)
    κοντινές ooc  42: επίπεδο 1 — πόσες κόβονται. Το «πέρασε» ΔΕΝ είναι αποτυχία (έχουν
                  εξ ορισμού σχετικό υλικό)· το επίπεδο 2 (τι ΑΠΑΝΤΑΕΙ) θέλει γέννηση
    συνομιλίες    12: κάλυψη μετά την αναδιατύπωση του follow-up (_rewrite_query της
                  παραγωγής)· τα 2 leak tests πρέπει να σωπάσουν
    πίνακες       16 (golden_tables, προστέθηκε 28/9): ήρθε η ΣΕΛΙΔΑ με τον πίνακα (doc+page
                  του σετ) + κάλυψη λέξεων. Οι ΤΙΜΕΣ της απάντησης -> scoreboard_answers --set tables
    άλλα πεδία    25 (golden_test_domains, προστέθηκε 28/9): 20 με απάντηση σε 2 PDF καρδιολογίας/
                  ηφαιστειολογίας + 5 ooc, σε ΔΙΚΟ ΤΟΥΣ store (/tmp/scoreboard_domains_chroma, 100
                  chunks). Σελίδα-στόχος (paper+page), κάλυψη, φύλακας, ooc — και ΠΟΙΟ πεδίο
                  γράφτηκε στο prompt μετάφρασης (διαβάζεται από το ίδιο το prompt).
    ΔΕΝ είναι εδώ: απαντήσεις/κριτής (-> scoreboard_answers.py).

ΠΑΓΩΜΑ ΤΟΥ GEMINI:
    Κάθε generate_once (μετάφραση, αναδιατύπωση corrective, αναδιατύπωση συνομιλίας) σώζεται
    στο runs/scoreboard_gemini.json με κλειδί ΟΛΟΚΛΗΡΟ το prompt. 2ο τρέξιμο = ίδιες
    απαντήσεις -> ό,τι αλλάζει ανάμεσα σε δύο τρεξίματα οφείλεται στο ΣΥΣΤΗΜΑ. Αν αλλάξει το
    prompt, το γλωσσάρι ή το domain, αλλάζει το κλειδί -> νέα κλήση ΑΥΤΟΜΑΤΑ (διορθώνει την
    παγίδα «το translation cache έχει key την ΕΡΩΤΗΣΗ, όχι το prompt»).
    Το cache μεταφράσεων της παραγωγής ΔΕΝ χρησιμοποιείται ούτε γράφεται.
    --fresh: αγνοεί το σωσμένο — ΜΟΝΟ για να μετρηθεί η διακύμανση του ίδιου του Gemini.
    Ο χρόνος ανάκτησης συγκρίνεται ΜΟΝΟ ανάμεσα σε παγωμένα τρεξίματα (νέα κλήση = +χρόνος Gemini).

ΕΛΕΓΧΟΣ, ΓΡΑΜΜΕΝΟΣ ΠΡΙΝ ΤΟ 1ο ΤΡΕΞΙΜΟ (28/9/2026) — πρέπει να αναπαραχθούν:
    φύλακας        59/61 · κενό −1.31 · κόβει με απάντηση q025, q059   (runs/gate_margin_now.csv)
    δύσκολο νέο    1ο πέρασμα κόβει 8 · σώζει ο corrective 6 (με υλικό) · με υλικό 52 · χωρίς 6 ·
                   σιωπή 1                                                 (runs/hard_new_eval.csv)
    δύσκολο παλιό  με υλικό 11 · χωρίς 3 · σιωπή 2                         (ίδιο αρχείο)
    κοντινές ooc   κόβονται 2/42                                           (runs/near_ooc_eval.csv)
    ooc κύριου     5/5 σιωπηλές
    Το 1ο τρέξιμο ξαναμεταφράζει (πάγωμα ανά prompt, όχι ανά ερώτηση)· ανεκτή απόκλιση ±1 ανά
    γραμμή ΜΟΝΟ αν εξηγείται από άλλη μετάφραση/αναδιατύπωση (το script τυπώνει πόσες
    μεταφράσεις διαφέρουν από τα προηγούμενα τρεξίματα). Μεγαλύτερη = σφάλμα του script.
    Άγνωστο (ΔΕΝ έχει μετρηθεί στο απομονωμένο store με τη σημερινή μετάφραση): το MRR του
    κύριου σετ — 10/8 ήταν 0.793 / 98.5%, στην παραγωγή 28/9 0.785 / 97.8% με ξένα αρχεία.

1ο ΤΡΕΞΙΜΟ (28/9, runs/scoreboard/baseline_draw1.json, 140 νέες κλήσεις Gemini, 0 σφάλματα):
    ✓ δύσκολο παλιό 11/3/2 · κοντινές 2/42 · ooc 5/5 · κενό −1.31
    ✗ φύλακας 60/61 (κόβει ΜΟΝΟ q059) · δύσκολο νέο 54/4/1, 1ο πέρασμα κόβει 7
    ΚΑΙ ΟΙ ΤΡΕΙΣ ΑΠΟΚΛΙΣΕΙΣ = ΑΛΛΗ ΜΕΤΑΦΡΑΣΗ (62/75 ίδιες με τα σωσμένα): q025 -> «cost per»
    (το κουτσουρεμένο του 25/9, περνάει +0.73), h126 / h145 (σημειωμένες ήδη ως ασταθείς στο
    probe_translation_prompt). Το script μετράει σωστά· το πάγωμα όμως κράτησε ΜΙΑ ΤΥΧΕΡΗ ΚΛΗΡΩΣΗ.
    Διόρθωση: στο πάγωμα η q025 πήρε τη συνηθισμένη μετάφραση (5/5 το πρωί) — σημειωμένο μέσα
    στο scoreboard_gemini.json («note»). h126/h145 έμειναν όπως κληρώθηκαν (δεν ξέρουμε ποια
    είναι η συνηθισμένη).
    ΠΡΟΒΛΕΨΗ 2ου ΤΡΕΞΙΜΑΤΟΣ (--compare baseline_draw1.json): 0 νέες κλήσεις Gemini · ΑΚΡΙΒΩΣ 1
    ερώτηση αλλάζει (q025 -> σιωπή) · φύλακας 59/61. Κάθε άλλη αλλαγή = μη ντετερμινισμός
    ΜΕΣΑ στο script ή στο σύστημα, και πρέπει να βρεθεί πριν τη Φάση 1.
    2ο ΤΡΕΞΙΜΟ (runs/scoreboard/baseline.json = Η ΒΑΣΗ ΣΥΓΚΡΙΣΗΣ ΤΗΣ ΦΑΣΗΣ 1): άλλαξε ΑΚΡΙΒΩΣ
    η q025 (0.73 -> −2.93, σιωπή)· οι άλλες 218 ταυτόσημες σε best1 ΚΑΙ σελίδες ✓. Φύλακας 59/61 ✓.
    1 νέα κλήση Gemini, όχι 0 ✗ (ο corrective της q025 με τη νέα μετάφραση — προβλέψιμο).
    Χρόνος 1.02 -> 0.66 s: στο 1ο τρέξιμο ο διάμεσος περιείχε τις κλήσεις μετάφρασης.
    ΝΕΟ ΕΥΡΗΜΑ: multi_hop νέες «και τα δύο papers» 7/29 (el 3/22 · en 4/7). ΟΧΙ σφάλμα μέτρησης
    (0 λέξεις λείπουν από τις σελίδες του store): σελίδα και από τα δύο papers έρχεται 54/58, η
    σελίδα-τεκμήριο 30/58. Υπόθεση «η μετάφραση πετάει το ζητούμενο» -> ΔΙΑΨΕΥΣΤΗΚΕ κατά το
    μεγαλύτερο μέρος (probe_multihop_lang: στα αγγλικά 19 -> 24/44 μόνο). Ο μηχανισμός: χάνεται
    το 2ο ΜΙΣΟ της ερώτησης (σελίδα του 1ου paper 19/29, του 2ου 11/29).

3ο ΤΡΕΞΙΜΟ — +ΠΙΝΑΚΕΣ ως 8ο σετ (28/9). Η βάση ξαναγράφεται ΜΕ τους πίνακες· το προηγούμενο
baseline.json κρατιέται αυτούσιο ως baseline_219.json.
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ: οι 219 παλιές ΤΑΥΤΟΣΗΜΕΣ (0 αλλαγές κατάστασης) · 16 νέες · 0 νέες
    κλήσεις Gemini (οι αγγλικές δεν μεταφράζονται· 1 κλήση ανά πίνακα που θα κοπεί, αναμένω 0-1)
    · ήρθε η σελίδα του πίνακα 14-16/16. Οποιαδήποτε αλλαγή στις 219 = μη ντετερμινισμός.
    ΑΠΟΤΕΛΕΣΜΑ: 219/219 ΤΑΥΤΟΣΗΜΕΣ (κατάσταση, best1, σελίδες) ✓ · 0 νέες κλήσεις ✓ · 16 νέες ✓ ·
    σελίδα του πίνακα 7/16 ✗ — ΕΝΩ οι απαντήσεις έβγαλαν 69/69 τιμές. Η αντίφαση έδειξε ΣΦΑΛΜΑ ΣΕΤ:
    το golden_tables.jsonl είχε σελίδες μετρημένες ΑΠΟ ΤΟ 0 (το dump_tables.py τυπώνει το pno του
    pymupdf), το store από το 1. Οι τιμές βρίσκονται στη σελίδα +1 σε 16/16 (έλεγχος με T.check σε
    κάθε σελίδα του store)· τα 7 «ήρθε» ήταν ΣΥΜΠΤΩΣΗ (είχε έρθει και η προηγούμενη σελίδα).
    Διορθώθηκε το σετ (page +1, τίποτα άλλο). Με σωστή σελίδα: ήρθε 16/16.

4ο ΤΡΕΞΙΜΟ — +ΑΛΛΑ ΠΕΔΙΑ ως 9ο σετ (28/9). Το προηγούμενο baseline.json -> baseline_235.json.
    docker compose exec backend python evaluation/scoreboard.py --label baseline \
        --compare evaluation/runs/scoreboard/baseline_235.json
    ΤΙ ΕΛΕΓΧΕΙ: ένα ανεβασμένο PDF παίρνει στο ingest domain="" (στατιστικό γλωσσάρι) -> το
    optimize_query πέφτει στο corpus_descriptor.json = «Cloud Computing and Distributed Systems».
    Μια ελληνική ερώτηση καρδιολογίας μεταφράζεται με πεδίο cloud ΚΑΙ με τους όρους του δικού της
    PDF. ΔΕΝ έχει μετρηθεί ποτέ: το 25/9 (domain_glossary_JK.csv, CD_constrained) είχε ΙΔΙΟ
    γλωσσάρι και ΙΔΙΑ περιοριστική οδηγία αλλά πεδίο «scientific research papers» -> κάλυψη 19/20
    (μόνο v3 0/2, −4.91 — σε ΟΛΕΣ τις συνθήκες, ανεξάρτητο από μετάφραση) · ooc 5/5 κομμένες στο
    1ο πέρασμα (πιο κοντά η o4 Βεζούβιος, −3.46). Μόνη διαφορά από εκείνη τη συνθήκη: το πεδίο.
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ (28/9/2026):
      οι 235 παλιές ΤΑΥΤΟΣΗΜΕΣ · 25 νέες · νέες κλήσεις Gemini 28-33 (25 μεταφράσεις + corrective
      στις κομμένες, ~6· ο corrective λέει «cloud computing» -> επιστρέφει το ερώτημα αμετάβλητο)
      πεδίο στο prompt: «Cloud Computing and Distributed Systems» ×25   (αλλιώς διάβασα λάθος τον κώδικα)
      αποτέλεσμα: με υλικό 18-20 · χωρίς 0-1 · σιωπή 0-2 (v3)
      σελίδα-στόχος 15-19/20 (η c6 έχει τις λέξεις της σε σ.1-2, ο στόχος είναι η σ.5)
      φύλακας 1ο πέρασμα 23-25/25 · ooc τελικά σιωπηλές 5/5 (ρίσκο: o4, 0.86 κάτω από το gate)
      ΥΠΟΘΕΣΗ: οι ιατρικοί/γεωλογικοί όροι έχουν ΜΙΑ μετάφραση -> το λάθος πεδίο αλλάζει λίγες
      διατυπώσεις, όχι αποτελέσματα.
      ΣΥΝΑΓΕΡΜΟΣ: ≥ 2 ερωτήσεις με απάντηση χάνουν υλικό ή ≥ 1 ooc περνάει -> το fallback στο cloud
      είναι ΠΡΑΓΜΑΤΙΚΟ σφάλμα παραγωγής για άλλα πεδία (το ingest θα έπρεπε να γράφει πεδίο).
    ΑΠΟΤΕΛΕΣΜΑ (28/9, runs/scoreboard/baseline.json = ΝΕΑ ΒΑΣΗ, 260 γραμμές):
      235/235 ΤΑΥΤΟΣΗΜΕΣ ✓ · 30 νέες κλήσεις ✓ · πεδίο «Cloud Computing…» ×25 ✓ · με υλικό 19 ·
      χωρίς 0 · σιωπή 1 (v3) ✓ · σελίδα-στόχος 18/20 ✓ (v3 κομμένη· v4 κάλυψη 2/2 από άλλες σελίδες)
      · φύλακας 23/25 ✓ · ooc 4/5 ✗ — η o4 (Βεζούβιος) ΠΕΡΝΑΕΙ ΤΟΝ ΦΥΛΑΚΑ στο 1ο πέρασμα:
      «Vesuvius eruption last time» −1.76 (25/9 με γενικό πεδίο: «Vesuvius eruptions» −3.46, κομμένη).
      Έξι στις επτά. Ο ΣΥΝΑΓΕΡΜΟΣ ΧΤΥΠΗΣΕ — αλλά η αιτία ΔΕΝ είναι αποδεδειγμένη: n=1, και η o4 είναι
      γνωστή ευαίσθητη στη διατύπωση (ήδη 3 μεταφράσεις: −3.46 κομμένη · «explosive activity» −1.60
      και «eruption rate» +0.49 περνούν). Χωρίς επανάληψη ανά πεδίο δεν ξεχωρίζει «φταίει το cloud»
      από «τυχαία διατύπωση».
      Ο χρόνος 0.73 -> 0.78 s είναι ΘΟΡΥΒΟΣ μηχανήματος: ίδιες ερωτήσεις, ίδιες σελίδες, ίδιοι βαθμοί
      (τα άλλα πεδία εξαιρούνται από τη γραμμή). Τρία τρεξίματα ίδιας δουλειάς: 0.66 / 0.73 / 0.78.
    ΕΠΙΒΕΒΑΙΩΣΗ ΑΙΤΙΑΣ (probe_vesuvius_domain.py): cloud 5/5 περνάει, γενικό 1/5 -> ΦΤΑΙΕΙ ΤΟ ΠΕΔΙΟ.
    Η απάντηση αρνείται σωστά (2ο στρώμα)· η ζημιά είναι στο 1ο.

5ο ΤΡΕΞΙΜΟ — ΔΙΟΡΘΩΣΗ: ο descriptor ισχύει ΜΟΝΟ για τα αρχεία που περιγράφει (28/9).
    ΑΛΛΑΓΗ: corpus_descriptor.json +"files" (τα 7 cloud) · corpus_glossary.scope_has_files ·
    ai_core: scope χωρίς ΚΑΝΕΝΑ αρχείο του descriptor -> _FALLBACK_DOMAIN («scientific research
    papers»). Μικτό scope -> όπως πριν (δεν μετρήθηκε). tests/test_scope_domain.py (6, χωρίς μοντέλα).
    docker compose exec backend python evaluation/scoreboard.py --label domain_fix \
        --compare evaluation/runs/scoreboard/baseline.json
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ:
      οι 235 του cloud ΤΑΥΤΟΣΗΜΕΣ με 0 νέες κλήσεις γι' αυτές (το prompt τους μένει byte προς byte
      ίδιο -> ίδιο κλειδί παγώματος) · άλλαξαν ΜΟΝΟ γραμμές των άλλων πεδίων
      πεδίο στο prompt: «scientific research papers» ×25 · νέες κλήσεις 25-31
      με υλικό 18-20 (25/9 με γενικό πεδίο: 19) · σελίδα-στόχος 16-19 · φύλακας 24/25 (μένει η v3)
      ooc 5/5 σιωπηλές (η o4 κόβεται ~4 στις 5· αν κληρωθεί το «… rate» περνάει — 1/5 στη μέτρηση)
    ΚΡΙΤΗΡΙΟ ΑΠΟΔΟΧΗΣ (γραμμένο πριν): cloud 235/235 ίδιες ΚΑΙ με υλικό ≥ 19 ΚΑΙ ooc ≥ 4/5.
      Αν αλλάξει ΕΣΤΩ ΜΙΑ γραμμή cloud -> η αλλαγή άγγιξε κάτι που δεν έπρεπε: ΑΠΟΣΥΡΕΤΑΙ.
    ΑΠΟΤΕΛΕΣΜΑ (runs/scoreboard/domain_fix.json): ΓΙΝΕΤΑΙ ΔΕΚΤΗ. Επτά στις επτά.
      cloud 235/235 ΤΑΥΤΟΣΗΜΕΣ ✓, 0 κλήσεις γι' αυτές ✓ (28 νέες = 25 μεταφράσεις + 3 corrective
      όπου άλλαξε η μετάφραση: v3 o4 o5) · πεδίο «scientific research papers» ×25 ✓ · με υλικό 19 ·
      χωρίς 0 · σιωπή 1 ✓ · σελίδα-στόχος 18/20 ✓ · φύλακας 24/25 ✓ · ooc 5/5 ✓ (o4 «Vesuvius
      eruptions» −3.46, κομμένη).
      MRR 0.858 -> 0.792: ΤΡΕΙΣ ερωτήσεις με ΑΛΛΗ διατύπωση (c7, v2, v10), ίδια κάλυψη ΚΑΙ ίδια σελίδα-
      στόχος και στις τρεις -> θόρυβος μετάφρασης σε n=20, όχι απώλεια υλικού. ΠΑΡΑΤΗΡΗΣΗ: με γενικό
      πεδίο οι μεταφράσεις βγαίνουν πιο ΛΙΤΕΣ («effusion rate vent», «Vesuvius eruptions») — το ίδιο
      που κόβει την o4 μπορεί σε άλλη ερώτηση να κόψει κομμάτι της. Εδώ δεν κόστισε υλικό.
      ΝΕΑ ΒΑΣΗ ΤΗΣ ΦΑΣΗΣ 1: baseline.json = αντίγραφο του domain_fix.json· το προηγούμενο (260 γραμμές,
      πριν τη διόρθωση) -> baseline_260_prefix.json.

6ο ΤΡΕΞΙΜΟ — ΦΑΣΗ 1.1: gte-reranker-modernbert-base (150M) στη θέση του MiniLM-L-12 (29/9).
    ΤΑΧΥΤΗΤΑ ΗΔΗ ΑΠΟΡΡΙΦΘΗΚΕ ΓΙΑ CPU (bench_reranker_latency.py: rerank 3.26 s vs 0.59, ανάκτηση ~3.3 s
    έναντι ορίου 1.2). Αυτό το τρέξιμο απαντάει ΑΛΛΗ ερώτηση: «ΣΕ GPU, όπου η ταχύτητα δεν μετράει,
    θα άξιζε;». Η ποιότητα είναι ίδια σε CPU/GPU (ίδιες πράξεις).
    ΡΥΘΜΙΣΕΙΣ: ai_core φορτώνει τον reranker με activation Identity (29/9· στο MiniLM 300 ζευγάρια
    max|Δ| = 0.0 — χωρίς αυτό το gte δίνει sigmoid και ο φύλακας περνάει ΤΑ ΠΑΝΤΑ).
      φύλακας 1.595 = μέσο του καλύτερου κατωφλίου στο gate_margin_gte.csv (q050 1.51 | q025 1.68)
      corrective −99 στο τρέξιμο -> το κατώφλι του ορίζεται ΕΚ ΤΩΝ ΥΣΤΕΡΩΝ με τον κανόνα του
      rethreshold_corrective.py (ίδιος κανόνας ξαναβγάζει στο MiniLM ακριβώς το −3.8: ΕΛΕΓΧΘΗΚΕ).
    docker compose exec -e RERANKER_MODEL=Alibaba-NLP/gte-reranker-modernbert-base \
        -e MIN_RERANK_SCORE=1.595 -e CORRECTIVE_MIN_SCORE=-99 backend \
        python evaluation/scoreboard.py --label gte_raw
    docker compose exec backend python evaluation/rethreshold_corrective.py \
        evaluation/runs/scoreboard/gte_raw.json --out gte --compare evaluation/runs/scoreboard/baseline.json
    Ο φύλακας βαθμονομήθηκε ΜΟΝΟ στις 61 του cloud -> κοντινές ooc και άλλα πεδία = ΕΚΤΟΣ ΔΕΙΓΜΑΤΟΣ.
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ:
      φύλακας 60/61 (κόβει μόνο q059 — γνωστό από gate_margin_gte) · ooc κύριου 5/5 · leak 2/2
      κύριο MRR 0.785 -> 0.80-0.83 (σωστό chunk 1ο: 41 -> 46/56) · κάλυψη 96-98% (ήταν 97.0)
      multi_hop νέες: και τα δύο 7/29 -> 7-10 · κάλυψη 54.3% -> 52-60%
      δύσκολο παλιό με υλικό 11 -> 10-13 · δύσκολο νέο με υλικό 54 -> 52-56, χωρίς 4 -> 3-6
      κοντινές κόβονται 2/42 -> 0-4 · πίνακες 15-16/16 · άλλα πεδία ooc 4-5/5 (το κατώφλι δεν τα είδε)
      νέες κλήσεις Gemini 5-25 (μόνο αναδιατυπώσεις ερωτήσεων που κόβονται για πρώτη φορά)
      ΣΥΝΟΛΙΚΑ: λίγο καλύτερη κατάταξη, ΧΩΡΙΣ ξεκάθαρο κέρδος στα σετ αποτυχιών.
    ΚΡΙΤΗΡΙΟ «ΑΞΙΖΕΙ ΣΕ GPU» (γραμμένο πριν) — ΟΛΑ:
      (1) καμία νέα διαρροή: ooc κύριου 5/5 · leak tests 2/2 · ooc άλλων πεδίων 5/5
      (2) καμία πτώση στα σετ-ταβάνι: κύρια κάλυψη ≥ 96% · πίνακες 16/16 · συνομιλίες 100%
      (3) βελτίωση σε ≥ 2 από τα 4 σημεία αποτυχίας, χωρίς χειροτέρευση στα άλλα:
          φύλακας (≥ 60/61) · multi_hop νέες «και τα δύο» ≥ 10/29 · δύσκολο νέο χωρίς+σιωπή ≤ 3 (ήταν 5)
          · δύσκολο παλιό με υλικό ≥ 12 (ήταν 11)
      Ακόμα και αν περάσει: n μικρό -> ΕΝΔΕΙΞΗ, όχι απόδειξη (bootstrap πριν γραφτεί ως κέρδος).
    ΑΠΟΤΕΛΕΣΜΑ (runs/scoreboard/gte.json· corrective με τον κανόνα = 1.57· 21 νέες κλήσεις Gemini):
      ΔΕΝ ΠΕΡΝΑΕΙ ΤΟ ΚΡΙΤΗΡΙΟ. (1) ✗ ο Βεζούβιος (o4, άλλα πεδία) ΠΕΡΝΑΕΙ τον φύλακα: 1.83 > 1.595 —
      το κατώφλι του cloud ΔΕΝ μεταφέρεται σε άλλο σώμα. (2) ✓ κάλυψη 97.0% · πίνακες 16/16 ·
      συνομιλίες 100%. (3) ✓ φύλακας 60/61 · ✓ «και τα δύο» 11/29 · ✗ δύσκολο νέο χωρίς+σιωπή 4
      (όχι ≤3) · ✗ δύσκολο παλιό: με υλικό 11 ΙΔΙΟ, αλλά χωρίς 3 -> 5: οι h009/h012 (σιωπή, σωστή για
      το h012 που δεν έχει αναφορικό) περνάνε τώρα τον φύλακα με 0 υλικό (2.09 / 1.84).
      ΤΟ ΚΕΡΔΟΣ ΕΙΝΑΙ ΠΡΑΓΜΑΤΙΚΟ ΚΑΙ ΜΕΓΑΛΟ ΣΤΗΝ ΚΑΤΑΤΑΞΗ (ζευγαρωτό bootstrap, 10.000, seed 42):
        κύριο MRR 0.785 -> 0.874  Δ +0.089 CI [+0.025, +0.160]  ΑΠΟΔΕΙΚΝΥΕΤΑΙ (15 καλύτερα / 6 χειρότερα)
        κύριο nDCG 0.795 -> 0.863 Δ +0.068 CI [+0.016, +0.130]  ΑΠΟΔΕΙΚΝΥΕΤΑΙ
        multi_hop 40 κάλυψη       Δ +0.075 CI [+0.000, +0.158]  στο όριο (7 καλύτερα / 2 χειρότερα)
        multi_hop νέες «και τα δύο» 7 -> 11/29 (el 3 -> 7)      CI [+0.000, +0.310] στο όριο
        δύσκολα 75 κάλυψη          Δ +0.027 CI [−0.027, +0.080] θόρυβος
        άλλα πεδία: σελίδα-στόχος 18 -> 20/20 · MRR 0.792 -> 0.909 · v3 λύνεται (2/2)
      Η ΠΡΩΤΗ βελτίωση του project που αποδεικνύεται στο κύριο σετ (το L-12 δεν αποδεικνυόταν).
      ΤΟ ΜΟΤΙΒΟ: καλύτερη ΚΑΤΑΤΑΞΗ, χειρότερο σήμα «δεν υπάρχει εδώ». Κλίμακα συμπιεσμένη (σχεδόν όλα
      στο 1-3) -> σωστές και άσχετες πολύ κοντά· περισσότερες «εύλογες αλλά λάθος» περνάνε (h009, h012,
      o4). Ίδιο μοτίβο με το bge-reranker-base τον Αύγουστο. Οι κοντινές ooc κόβονται 2 -> 9/42.
      Πρόβλεψη 10/15: ✗ MRR (0.874, υποτίμησα) · ✗ «και τα δύο» (11) · ✗ κάλυψη multi_hop (62.4) ·
      ✗ κοντινές (9) · ✗ «χωρίς ξεκάθαρο κέρδος» — ΥΠΟΤΙΜΗΣΑ ΤΗΝ ΚΑΤΑΤΑΞΗ, όχι τον κίνδυνο.

7ο ΤΡΕΞΙΜΟ — ΦΑΣΗ 1.2: Qwen3-Embedding-0.6B στη θέση του bge-m3 (29/9). Reranker ΙΔΙΟΣ (MiniLM),
    κατώφλια ΙΔΙΑ (−2.6 / −3.8: ίδιος reranker = ίδια κλίμακα) -> η ΜΟΝΗ μεταβλητή είναι ποια 30
    κομμάτια φέρνει το dense σκέλος. Ταχύτητα: bench_embedder.py (ερώτηση +69 ms μόνο την 1η φορά,
    ingest 1.84× — ο κανόνας 1.25× έπεσε, αποφασίστηκε ΜΑΖΙ να συνεχίσουμε: η ανάκτηση μένει ≤1.2 s).
    Διακόπτης: evaluation/embedder_switch.py (ΚΑΜΙΑ αλλαγή στην παραγωγή· ΔΙΚΑ ΤΟΥ stores με κατάληξη
    _qwen3-embedding-0.6b, χτίζονται την 1η φορά: ~11 + ~3 λεπτά).
    docker compose exec -e EVAL_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B backend python \
        evaluation/scoreboard.py --label qwen3 --compare evaluation/runs/scoreboard/baseline.json
    ⚠️ Η γραμμή «χρόνος ανάκτησης» ΔΕΝ συγκρίνεται: στη βάση τα διανύσματα ερωτήσεων ήταν στο cache
    της παραγωγής, εδώ φτιάχνονται όλα από την αρχή (+~0.25 s). Ο χρόνος μετρήθηκε στο bench_embedder.
    ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ:
      φύλακας 59/61 · ooc κύριου 5/5 · leak 2/2 · άλλα πεδία ooc 5/5 (ίδιος reranker, ίδιο κατώφλι)
      κύριο MRR 0.785 -> 0.77-0.82 (σετ-ταβάνι) · κάλυψη 96-98%
      multi_hop νέες «και τα δύο» 7 -> 8-12/29 (dense top-30: 12 -> 17) · κάλυψη 54.3% -> 55-63%
      δύσκολο παλιό με υλικό 11 -> 10-13 · δύσκολο νέο με υλικό 54 -> 53-57, χωρίς 4 -> 2-5
      κοντινές κόβονται 2/42 -> 1-4 · πίνακες 16/16 · συνομιλίες 100% · νέες κλήσεις Gemini 0-15
    ΚΡΙΤΗΡΙΟ «ΚΡΑΤΙΕΤΑΙ» (ΙΔΙΟ με του 6ου) — ΟΛΑ:
      (1) καμία νέα διαρροή: ooc κύριου 5/5 · leak tests 2/2 · ooc άλλων πεδίων 5/5
      (2) καμία πτώση στα σετ-ταβάνι: κύρια κάλυψη ≥ 96% · πίνακες 16/16 · συνομιλίες 100%
      (3) βελτίωση σε ≥ 2 από τα 4 σημεία αποτυχίας, χωρίς χειροτέρευση στα άλλα:
          φύλακας (≥ 60/61) · multi_hop νέες «και τα δύο» ≥ 10/29 · δύσκολο νέο χωρίς+σιωπή ≤ 3 (ήταν 5)
          · δύσκολο παλιό με υλικό ≥ 12 (ήταν 11)
      Αν περάσει: bootstrap πριν γραφτεί ως κέρδος· και ΜΕΤΑ κριτής απαντήσεων (θα ρωτηθεί το κόστος).
    ΑΠΟΤΕΛΕΣΜΑ (runs/scoreboard/qwen3.json· 0 νέες κλήσεις Gemini, 170 από το πάγωμα): ΑΠΟΡΡΙΠΤΕΤΑΙ.
      (1) ✗ ΔΥΟ ΔΙΑΡΡΟΕΣ: q050 («GPU tensor cores») μέσω corrective — ΙΔΙΑ αναδιατύπωση, αλλά το
      Qwen3 έφερε άλλα υποψήφια: best2 −5.29 -> −3.67 (0.13 πάνω από το −3.8)· o4 (Βεζούβιος) από τον
      φύλακα: best1 −3.46 -> −2.52 (0.08 πάνω από το −2.6). (2) ✓ κάλυψη 97.0 · πίνακες 16/16 · συνομιλίες 100.
      (3) 0/4: φύλακας 59/61 · «και τα δύο» 9/29 (<10) · δύσκολο νέο χωρίς+σιωπή 6 (ήταν 5, ΧΕΙΡΟΤΕΡΑ)
      · δύσκολο παλιό με υλικό 10 (ήταν 11, ΧΕΙΡΟΤΕΡΑ: h008 3/3 -> 0/3).
      bootstrap: κύριο MRR 0.785 -> 0.799 Δ +0.013 CI [−0.008, +0.043] · multi_hop 40 κάλυψη Δ +0.019
      CI [−0.031, +0.073] · δύσκολα 75 Δ −0.013 CI [−0.071, +0.040] — ΟΛΑ ΘΟΡΥΒΟΣ.
      Σελίδες άλλαξαν σε 221/260 ερωτήσεις — και ΚΑΜΙΑ μετρική δεν κουνήθηκε πέρα από τον θόρυβο.
      ΤΟ ΚΕΡΔΟΣ ΤΟΥ DENSE ΕΞΑΤΜΙΣΤΗΚΕ: στο top-30 «και τα δύο» 12 -> 17 (bench_embedder), στις 8 σελίδες
      7 -> 9. Το RRF + reranker αναδιατάσσουν — ίδιο μάθημα με το «καμία μετρική ενδιάμεσου σταδίου δεν
      προβλέπει το τελικό prompt».
      ΤΟ ΜΑΘΗΜΑ: ΙΔΙΟΣ reranker + ΙΔΙΟ κατώφλι ΔΕΝ σημαίνει ίδιος φύλακας. Ο φύλακας κρίνει τον
      καλύτερο βαθμό ΑΝΑΜΕΣΑ στα υποψήφια· άλλος embedder -> άλλα υποψήφια -> άλλος καλύτερος βαθμός.
      Οι οριακές (0.08, 0.13) γυρίζουν. Κάθε αλλαγή ΑΝΑΝΤΗ του φύλακα (μετάφραση, embedder, υποψήφια)
      αλλάζει την είσοδό του — όπως η παλινδρόμηση μετάφρασης 16-18/8.
      Πρόβλεψη 13/15 — τα δύο λάθη ΗΤΑΝ οι διαρροές («ίδιος reranker -> ίδιος φύλακας»).

ΠΡΟΣΟΧΗ: το store (/tmp/eval_near_ooc_chroma) είναι ΚΟΙΝΟ με eval_near_ooc / eval_hard_new /
gate_margin_isolated. Η κλειδαριά εδώ προστατεύει μόνο από 2ο scoreboard — μην τρέχεις
κανένα από αυτά ταυτόχρονα (ΠΑΓΙΔΕΣ, 28/9).

    # δοκιμή άκρο-έως-άκρο, 2 ερωτήσεις ανά σετ (δεν σώζει πάγωμα ούτε αποτέλεσμα):
    docker compose exec backend python evaluation/scoreboard.py --limit 2
    # πλήρες (~10-15 λεπτά):
    docker compose exec backend python evaluation/scoreboard.py --label baseline
    # μετά από αλλαγή (π.χ. reranker), δίπλα-δίπλα με το baseline:
    docker compose exec -e RERANKER_MODEL=... backend python evaluation/scoreboard.py \
        --label nea_allagi --compare evaluation/runs/scoreboard/baseline.json
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import embedder_switch
import eval_near_ooc as E

import gemini_rest

HERE = E.HERE
RUNS = os.path.join(HERE, "runs")
OUT_DIR = os.path.join(RUNS, "scoreboard")
FROZEN_PATH = os.path.join(RUNS, "scoreboard_gemini.json")
LOCK = "/tmp/scoreboard.lock"  # noqa: S108  εφήμερο, μέσα στο container
EXPECTED_CHUNKS = 418
OLD_TRANS = ["hard_translations.json", "near_ooc_translations.json", "gate_margin_translations.json"]
SETS = [
    ("main", "golden_set_50.jsonl"),
    ("mh_old", "golden_multihop_new.jsonl"),
    ("mh_new", "golden_multihop_v2.jsonl"),
    ("hard_old", "golden_hard_paraphrase.jsonl"),
    ("hard_new", "golden_hard_new.jsonl"),
    ("near_ooc", "golden_near_ooc.jsonl"),
    ("conv", "golden_conversations.jsonl"),
    ("tables", "golden_tables.jsonl"),      # 28/9: 8ο σετ — βλ. docstring «ΠΙΝΑΚΕΣ»
    # 28/9: 9ο σετ, ΑΛΛΟ ΣΩΜΑ (2 PDF καρδιολογίας/ηφαιστειολογίας) σε ΔΙΚΟ ΤΟΥ store.
    # ΠΡΕΠΕΙ να μείνει ΤΕΛΕΥΤΑΙΟ: το run() αλλάζει το ai_core.collection πριν από αυτό.
    ("domains", "golden_test_domains.jsonl"),
]
DOMAINS_DB = "/tmp/scoreboard_domains_chroma"  # noqa: S108  εφήμερο, μέσα στο container
DOMAINS_PAPERS = ["cureus-0015-00000046486.pdf", "s41598-017-03833-3.pdf"]
DOMAINS_CHUNKS = 100      # 63 + 37, μετρημένο με το split του ingest_pdf χωρίς μοντέλα (28/9)
_DOMAIN_RE = re.compile(r"The corpus is about: (.*?)\. Translate the user")
SHORT = {"cut": "σιωπή", "passed_gate": "φύλακας", "passed_corrective": "corrective"}
CONTROL = [
    ("φύλακας · σωστά στο 1ο πέρασμα", "59/61"),
    ("φύλακας · κενό", "−1.31"),
    ("φύλακας · κόβει με απάντηση", "q025, q059"),
    ("δύσκολο νέο · 1ο πέρασμα κόβει / σώζει ο corrective", "8 / 6"),
    ("δύσκολο νέο · αποτέλεσμα", "με υλικό 52 · χωρίς 6 · σιωπή 1"),
    ("δύσκολο παλιό · αποτέλεσμα", "με υλικό 11 · χωρίς 3 · σιωπή 2"),
    ("κοντινές ooc · κόβονται (επίπεδο 1)", "2/42"),
    ("ooc κύριου · τελικά σιωπηλές", "5/5"),
]


class Frozen:
    """Πάγωμα του generate_once με κλειδί ΟΛΟΚΛΗΡΟ το prompt. Τυλίγει τον κατάσκοπο του
    eval_near_ooc, ώστε το E.retrieve να βρίσκει την αναδιατύπωση και στα παγωμένα."""

    def __init__(self, path: str, fresh: bool):
        self.path = path
        self.data: dict = {}
        if not fresh and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.hits = self.misses = 0
        self.errors: list[str] = []
        self.last_domain = ""   # «The corpus is about: X» του τελευταίου prompt ΜΕΤΑΦΡΑΣΗΣ

    def install(self) -> None:
        inner = gemini_rest.generate_once

        async def frozen(prompt, **kw):
            m = _DOMAIN_RE.search(prompt)
            if m:
                self.last_domain = m.group(1)
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if key in self.data:
                self.hits += 1
                text = self.data[key]["out"]
                E._prompts.append((prompt[:60], text.strip(" \"'\n")))
                return text
            try:
                text = await inner(prompt, **kw)
            except Exception as e:
                # Το optimize_query ΚΑΤΑΠΙΝΕΙ το σφάλμα και ψάχνει με την αμετάφραστη
                # ερώτηση -> σιωπηλά λάθος νούμερα. Καταγράφεται και ακυρώνει το τρέξιμο.
                self.errors.append(f"{type(e).__name__}: {e}"[:200])
                raise
            self.misses += 1
            self.data[key] = {"tail": prompt[-200:], "out": text}
            return text

        gemini_rest.generate_once = frozen

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)


def lang_of(t: dict, question: str) -> str:
    if t.get("lang"):
        return t["lang"]
    return "el" if any("Ͱ" <= ch <= "Ͽ" for ch in question) else "en"


def both_docs(t: dict, pages) -> int | None:
    """1 αν για ΚΑΘΕ paper της ερώτησης βρέθηκε λέξη του μέσα σε σελίδα ΤΟΥ ΙΔΙΟΥ paper."""
    by = t.get("keywords_by_doc") or {}
    if not by:
        return None
    ok = 0
    for doc, kws in by.items():
        texts = [x.lower() for x, m in pages if m.get("file_name") == doc]
        ok += any(k.lower() in x for k in kws for x in texts)
    return int(ok == len(by))


def state(r: dict) -> str:
    """Σύντομη κατάσταση ανά ερώτηση — ό,τι αλλάζει εδώ ανάμεσα σε δύο τρεξίματα, τυπώνεται."""
    s = f"{SHORT[r['outcome']]} {r['cov']}/{r['n_kw']}"
    if r.get("both") is not None:
        s += " δύο:" + ("ναι" if r["both"] else "όχι")
    return s


def make_row(set_key: str, t: dict, question: str, res, ai_core, mrr_fn, ndcg_fn) -> dict:
    pages, b1, b2, rw, outcome, dt = res
    kws = t.get("keywords") or []
    doc = t.get("doc") or (t.get("paper") if t.get("paper") != "-" else None)
    texts = [x for x, _m in pages]
    mrrs = [mrr_fn(k, texts) for k in kws] if texts else [0.0] * len(kws)
    ndcgs = [ndcg_fn(k, texts, k=3) for k in kws] if texts else [0.0] * len(kws)
    row = {
        "key": f"{set_key}:{t['id']}", "set": set_key, "id": t["id"],
        "category": t.get("category", ""), "lang": lang_of(t, question),
        "question": question, "translation": ai_core._translation_cache.get(question, ""),
        "outcome": outcome, "best1": None if b1 is None else round(b1, 2),
        "best2": None if b2 is None else round(b2, 2), "rewrite": rw,
        "n_pages": len(pages), "n_kw": len(kws), "cov": sum(m > 0 for m in mrrs),
        "mrr": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
        "ndcg": round(statistics.mean(ndcgs), 4) if ndcgs else 0.0,
        "both": both_docs(t, pages), "set_error": int(bool(t.get("set_error"))), "doc": doc,
        # ήρθε η ΣΕΛΙΔΑ-ΣΤΟΧΟΣ; (μόνο σετ με doc/paper + page: πίνακες, άλλα πεδία)
        "target": (int(any(m.get("file_name") == doc and str(m.get("page")) == str(t["page"])
                           for _x, m in pages)) if doc and t.get("page") else None),
        "leak_test": int(bool(t.get("leak_test"))),
        "pages": ";".join(f"{m.get('file_name', '?')}:{m.get('page')}" for _x, m in pages),
        "sec": round(dt, 2),
    }
    row["state"] = state(row)
    return row


def summarize(rows: list[dict], gate: float) -> dict:
    """{όνομα γραμμής: κείμενο}. Κάθε σετ με τον τρόπο του script που το μετρούσε ως τώρα."""
    S: dict = {}

    def of(k):
        return [r for r in rows if r["set"] == k]

    def b1(r):
        return float("-inf") if r["best1"] is None else r["best1"]

    def mean_cov(rs):
        return 100 * statistics.mean(r["cov"] / r["n_kw"] for r in rs if r["n_kw"])

    main = of("main")
    m_in = [r for r in main if r["category"] != "out_of_corpus"]
    m_ooc = [r for r in main if r["category"] == "out_of_corpus"]
    if m_in:
        S["κύριο · MRR"] = f"{statistics.mean(r['mrr'] for r in m_in):.3f}  (n={len(m_in)})"
        S["κύριο · nDCG@3"] = f"{statistics.mean(r['ndcg'] for r in m_in):.3f}"
        S["κύριο · κάλυψη λέξεων"] = f"{mean_cov(m_in):.1f}%"
        for c in sorted({r["category"] for r in m_in}):
            sub = [r["mrr"] for r in m_in if r["category"] == c]
            S[f"κύριο · MRR {c}"] = f"{statistics.mean(sub):.3f}  (n={len(sub)})"

    g_in = m_in + of("mh_old")
    if g_in and m_ooc:
        ok_in = [r for r in g_in if b1(r) >= gate]
        ok_ooc = [r for r in m_ooc if b1(r) < gate]
        lo, hi = min(b1(r) for r in g_in), max(b1(r) for r in m_ooc)
        S["φύλακας · σωστά στο 1ο πέρασμα"] = f"{len(ok_in) + len(ok_ooc)}/{len(g_in) + len(m_ooc)}"
        S["φύλακας · κενό"] = f"{lo - hi:+.2f}  (σωστή {lo:.2f} · άσχετη {hi:.2f})"
        S["φύλακας · κόβει με απάντηση"] = ", ".join(
            sorted(r["id"] for r in g_in if b1(r) < gate)) or "καμία"
        S["ooc κύριου · τελικά σιωπηλές"] = f"{sum(not r['n_pages'] for r in m_ooc)}/{len(m_ooc)}"

    for k, name in (("mh_old", "multi_hop παλιές"), ("mh_new", "multi_hop νέες")):
        rs = of(k)
        if rs:
            S[f"{name} · κάλυψη"] = f"{mean_cov(rs):.1f}%  (n={len(rs)})"
    both = [r for r in of("mh_new") if r["both"] is not None]
    if both:
        parts = " · ".join(f"{g} {sum(r['both'] for r in both if r['lang'] == g)}/"
                           f"{sum(r['lang'] == g for r in both)}"
                           for g in ("el", "en") if any(r["lang"] == g for r in both))
        S["multi_hop νέες · και τα δύο papers"] = f"{sum(r['both'] for r in both)}/{len(both)}  ({parts})"

    for k, name in (("hard_old", "δύσκολο παλιό"), ("hard_new", "δύσκολο νέο")):
        rs = of(k)
        if not rs:
            continue
        v = Counter("σιωπή" if not r["n_pages"] else "με υλικό" if r["cov"] else "χωρίς"
                    for r in rs)
        S[f"{name} · αποτέλεσμα"] = (f"με υλικό {v['με υλικό']} · χωρίς {v['χωρίς']} · "
                                     f"σιωπή {v['σιωπή']}  (n={len(rs)})")
        S[f"{name} · 1ο πέρασμα κόβει / σώζει ο corrective"] = (
            f"{sum(b1(r) < gate for r in rs)} / "
            f"{sum(r['outcome'] == 'passed_corrective' and r['cov'] > 0 for r in rs)}")
        errs = [r["id"] for r in rs if r["set_error"]]
        if errs:
            S[f"{name} · σφάλματα σετ (μετράνε μέσα)"] = ", ".join(errs)

    near = of("near_ooc")
    if near:
        S["κοντινές ooc · κόβονται (επίπεδο 1)"] = (
            f"{sum(not r['n_pages'] for r in near)}/{len(near)}  "
            f"(μέσω corrective πέρασαν {sum(r['outcome'] == 'passed_corrective' for r in near)})")

    conv = of("conv")
    c_in = [r for r in conv if not r["leak_test"]]
    c_leak = [r for r in conv if r["leak_test"]]
    if c_in:
        tot = sum(r["n_kw"] for r in c_in)
        S["συνομιλίες · κάλυψη"] = f"{100 * sum(r['cov'] for r in c_in) / tot:.1f}%  (n={len(c_in)})"
    if c_leak:
        S["συνομιλίες · leak tests σιωπηλά"] = f"{sum(not r['n_pages'] for r in c_leak)}/{len(c_leak)}"

    tab = of("tables")
    if tab:
        S["πίνακες · ήρθε η σελίδα του πίνακα"] = (
            f"{sum(r['target'] or 0 for r in tab)}/{len(tab)}  (σιωπή {sum(not r['n_pages'] for r in tab)})")
        S["πίνακες · κάλυψη λέξεων"] = f"{mean_cov(tab):.1f}%"

    dom = of("domains")
    d_in = [r for r in dom if r["category"] != "out_of_corpus"]
    d_ooc = [r for r in dom if r["category"] == "out_of_corpus"]
    if dom:
        seen = Counter(r["tr_domain"] for r in dom if r.get("tr_domain"))
        S["άλλα πεδία · πεδίο στο prompt μετάφρασης"] = (
            " · ".join(f"«{d}» ×{n}" for d, n in seen.most_common()) or "καμία μετάφραση")
    if d_in:
        v = Counter("σιωπή" if not r["n_pages"] else "με υλικό" if r["cov"] else "χωρίς" for r in d_in)
        S["άλλα πεδία · αποτέλεσμα"] = (f"με υλικό {v['με υλικό']} · χωρίς {v['χωρίς']} · "
                                        f"σιωπή {v['σιωπή']}  (n={len(d_in)})")
        parts = " · ".join(f"{name} {sum(r['target'] or 0 for r in d_in if r['doc'] == p)}/"
                           f"{sum(r['doc'] == p for r in d_in)}"
                           for p, name in zip(DOMAINS_PAPERS, ("καρδιολογία", "ηφαιστειολογία")))
        S["άλλα πεδία · ήρθε η σελίδα-στόχος"] = f"{sum(r['target'] or 0 for r in d_in)}/{len(d_in)}  ({parts})"
        S["άλλα πεδία · κάλυψη λέξεων · MRR"] = (f"{mean_cov(d_in):.1f}% · "
                                                f"{statistics.mean(r['mrr'] for r in d_in):.3f}")
    if d_in and d_ooc:
        lo, hi = min(b1(r) for r in d_in), max(b1(r) for r in d_ooc)
        S["άλλα πεδία · φύλακας 1ο πέρασμα"] = (
            f"{sum(b1(r) >= gate for r in d_in) + sum(b1(r) < gate for r in d_ooc)}/{len(dom)} · "
            f"κενό {lo - hi:+.2f} · κόβει με απάντηση: "
            + (", ".join(r["id"] for r in d_in if b1(r) < gate) or "καμία"))
        leaked = [r["id"] for r in d_ooc if r["n_pages"]]
        S["άλλα πεδία · ooc τελικά σιωπηλές"] = (f"{len(d_ooc) - len(leaked)}/{len(d_ooc)}"
                                                 + (f"  (πέρασαν: {', '.join(leaked)})" if leaked else ""))

    # ΟΧΙ τα άλλα πεδία: άλλο store (100 chunks αντί 418) -> θα μετακινούσε τον διάμεσο
    fast = [r["sec"] for r in rows if r["outcome"] == "passed_gate" and r["set"] not in ("conv", "domains")]
    if fast:
        S["χρόνος ανάκτησης · διάμεσος (χωρίς corrective)"] = f"{statistics.median(fast):.2f} s"
    return S


def print_table(S: dict, prev: dict | None) -> None:
    w = max(len(k) for k in S) + 2
    print("\n" + "#" * 100)
    if prev:
        print(f"  {'':<{w}}{'ΠΡΙΝ':<46}ΤΩΡΑ")
        for k, v in S.items():
            old = prev.get(k, "—")
            mark = "  " if old == v else "* "
            print(f"{mark}{k:<{w}}{old:<46}{v}")
        for k in prev:
            if k not in S:
                print(f"* {k:<{w}}{prev[k]:<46}—")
    else:
        for k, v in S.items():
            print(f"{k:<{w}}{v}")
    print("#" * 100)


def print_flips(rows: list[dict], prev_rows: list[dict]) -> None:
    old = {r["key"]: r for r in prev_rows}
    flips = [(r, old[r["key"]]) for r in rows if r["key"] in old and old[r["key"]]["state"] != r["state"]]
    print(f"\nΕΡΩΤΗΣΕΙΣ ΠΟΥ ΑΛΛΑΞΑΝ ({len(flips)} από {len(rows)}):")
    for r, o in flips:
        print(f"  {r['set']:<9}{r['id']:<6}{o['state']:<26}-> {r['state']:<26}"
              f"best1 {o['best1']} -> {r['best1']}   {r['question'][:60]}")
    new = [r for r in rows if r["key"] not in old]
    if new:
        print(f"  ({len(new)} ερωτήσεις δεν υπήρχαν στο προηγούμενο τρέξιμο)")


def translation_drift(rows: list[dict], old_trans: dict) -> None:
    same, diff = 0, []
    for r in rows:
        q, t = r["question"], r["translation"]
        if r["set"] == "conv" or not t or q not in old_trans:
            continue
        if old_trans[q] == t:
            same += 1
        else:
            diff.append(r)
    print(f"\nΜεταφράσεις ίδιες με τα προηγούμενα τρεξίματα: {same}/{same + len(diff)}")
    for r in diff[:12]:
        print(f"  {r['set']:<9}{r['id']:<6}παλιά «{old_trans[r['question']][:55]}»\n"
              f"{'':<17}τώρα  «{r['translation'][:55]}»   [{r['state']}]")


def open_domains_store(ai_core, rebuild: bool = False) -> int:
    """Στρέφει το ai_core στο store ΑΛΛΩΝ ΠΕΔΙΩΝ (το χτίζει αν λείπει, ~2 λεπτά σε CPU, 0 κλήσεις).
    ΔΕΝ ξαναβάζει κατασκόπους (τους έβαλε το E.open_store — διπλοί θα χάλαγαν best1/prompts).
    Ingest με τον ΚΩΔΙΚΑ ΤΗΣ ΠΑΡΑΓΩΓΗΣ: στατιστικό γλωσσάρι, domain="" -> η μετάφραση πέφτει
    στο καθολικό domain, ό,τι ακριβώς θα γινόταν αν ένας χρήστης ανέβαζε αυτά τα PDF.
    Επιστρέφει τον αριθμό chunks."""
    import chromadb
    if rebuild and os.path.exists(DOMAINS_DB):
        shutil.rmtree(DOMAINS_DB)
    col = chromadb.PersistentClient(path=DOMAINS_DB).get_or_create_collection(
        name="scoreboard_domains", embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    ai_core.collection = col
    if col.count() == 0:
        print(f"Χτίσιμο store άλλων πεδίων: {len(DOMAINS_PAPERS)} PDF (~2 λεπτά σε CPU)...", flush=True)
        for i, name in enumerate(DOMAINS_PAPERS):
            ok = ai_core.ingest_pdf(os.path.join(HERE, "test_papers", name), name,
                                    user_id=E.TEST_USER, is_public=False, doc_id=900 + i)
            print(f"  {name}: {'OK' if ok else 'ΚΕΝΟ'}", flush=True)
    ai_core._bump_corpus_version()           # άλλο σώμα -> BM25/dense caches από την αρχή
    return col.count()


async def run(args) -> int:
    global DOMAINS_DB
    if os.getenv(embedder_switch.ENV):
        # Φάση 1.2: ο διακόπτης μπαίνει ΜΕΤΑ το import και ΠΡΙΝ ανοίξει το store -> ΔΙΚΑ ΤΟΥ stores
        print("Φόρτωση μοντέλων (40-60 s χωρίς έξοδο) — ΜΗΝ το διακόψεις...", flush=True)
        import ai_core as _ai
        suffix = embedder_switch.install(_ai)
        if suffix:
            E.TEST_DB = f"{E.TEST_DB}_{suffix}"
            DOMAINS_DB = f"{DOMAINS_DB}_{suffix}"
    ai_core, _near = E.open_store(args.rebuild)
    from eval_engine import calculate_mrr, calculate_ndcg  # ΜΕΤΑ το open_store: ίδιο ai_core

    n_chunks = ai_core.collection.count()
    if n_chunks != EXPECTED_CHUNKS:
        print(f"!! Το store έχει {n_chunks} chunks, περίμενα {EXPECTED_CHUNKS} — "
              f"ξανατρέξε με --rebuild (και βεβαιώσου ότι δεν τρέχει άλλο script)")
        return 1

    old_trans = dict(ai_core._translation_cache)
    for name in OLD_TRANS:
        p = os.path.join(RUNS, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                old_trans.update(json.load(f))
    ai_core._translation_cache.clear()          # μεταφράσεις ΜΟΝΟ μέσα από το πάγωμα
    ai_core.ENABLE_CORRECTIVE = True            # όπως η παραγωγή
    frozen = Frozen(FROZEN_PATH, args.fresh)
    frozen.install()
    gate = ai_core.MIN_RERANK_SCORE
    config = {"reranker": ai_core.RERANKER_MODEL, "embedder": ai_core.EMBED_MODEL_NAME,
              "gate": gate, "corrective": ai_core.CORRECTIVE_MIN_SCORE,
              "corrective_on": ai_core.ENABLE_CORRECTIVE, "chunks": n_chunks,
              "gemini_model": E.MODEL, "limit": args.limit}
    print(f"Ρυθμίσεις: {config} · παγωμένες κλήσεις Gemini: {len(frozen.data)}")

    only = set(args.only.split(",")) if args.only else None
    rows: list[dict] = []
    for key, fname in SETS:
        if only and key not in only:
            continue
        if key == "domains":
            n_dom = open_domains_store(ai_core, args.rebuild)
            config["domains_chunks"] = n_dom
            if n_dom != DOMAINS_CHUNKS:
                print(f"!! Το store άλλων πεδίων έχει {n_dom} chunks, περίμενα {DOMAINS_CHUNKS} — "
                      f"ξανατρέξε με --rebuild (και βεβαιώσου ότι δεν τρέχει άλλο script)")
                return 1
            print(f"Store άλλων πεδίων: {n_dom} chunks στο {DOMAINS_DB}", flush=True)
        tests = E.load_jsonl(os.path.join(HERE, fname))[:args.limit or None]
        print(f"\n===== {key}: {len(tests)} ερωτήσεις ({fname}) =====", flush=True)
        for t in tests:
            frozen.last_domain = ""
            question = t.get("question", "")
            if key == "conv":
                history = [{"role": "user", "content": t["turn1_question"]},
                           {"role": "assistant", "content": t["turn1_answer"]}]
                question = await ai_core._rewrite_query(t["followup"], history)
            t0 = time.perf_counter()
            pages, b1, b2, rw, outcome = await E.retrieve(ai_core, question)
            res = (pages, b1, b2, rw, outcome, time.perf_counter() - t0)
            row = make_row(key, t, question, res, ai_core, calculate_mrr, calculate_ndcg)
            row["tr_domain"] = frozen.last_domain    # "" = αγγλική ή ήδη μεταφρασμένη στη διεργασία
            if key == "conv":
                row["followup"] = t["followup"]
            rows.append(row)
            print(f"  {t['id']:<6}{row['state']:<26}best1 {E.fmt(b1):>6}  "
                  f"{row['sec']:>5.2f}s  {question[:60]}", flush=True)

    if not args.limit:
        frozen.save()
    S = summarize(rows, gate)

    prev = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            prev = json.load(f)
        print(f"\nΣύγκριση με: {args.compare}  ({prev.get('label')}, {prev.get('when')})")
        print(f"  τότε: {prev.get('config')}\n  τώρα: {config}")
    print_table(S, prev["summary"] if prev else None)
    if prev:
        print_flips(rows, prev["rows"])
    else:
        print("\nΕΛΕΓΧΟΣ (γραμμένος πριν το 1ο τρέξιμο — βλ. docstring):")
        for k, want in CONTROL:
            print(f"  {k:<52}αναμενόταν {want:<34}μετρήθηκε {S.get(k, '—')}")
    translation_drift(rows, old_trans)

    print(f"\nGemini: {frozen.misses} νέες κλήσεις · {frozen.hits} από το πάγωμα"
          f"{'' if args.limit else ' -> ' + FROZEN_PATH}")
    if frozen.errors:
        print(f"!! {len(frozen.errors)} κλήσεις Gemini ΑΠΕΤΥΧΑΝ (π.χ. {frozen.errors[0]}) — "
              f"οι αντίστοιχες ερωτήσεις έψαξαν ΑΜΕΤΑΦΡΑΣΤΕΣ. ΤΟ ΤΡΕΞΙΜΟ ΔΕΝ ΣΥΓΚΡΙΝΕΤΑΙ.")

    if args.limit:
        print("\n(--limit: δοκιμή — δεν σώθηκε ούτε αποτέλεσμα ούτε πάγωμα)")
        return 1 if frozen.errors else 0
    label = args.label or datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"label": label, "when": datetime.now().isoformat(timespec="minutes"),
           "config": config, "summary": S, "rows": rows,
           "gemini": {"new": frozen.misses, "frozen": frozen.hits, "errors": frozen.errors}}
    jpath, cpath = os.path.join(OUT_DIR, f"{label}.json"), os.path.join(OUT_DIR, f"{label}.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"\nΣώθηκε: {jpath}\n        {cpath}")
    return 1 if frozen.errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Όλα τα σετ, ένας πίνακας, απομονωμένο store.")
    ap.add_argument("--label", default="", help="όνομα τρεξίματος (default: ημερομηνία-ώρα)")
    ap.add_argument("--compare", default="", help="προηγούμενο runs/scoreboard/<label>.json")
    ap.add_argument("--limit", type=int, default=0, help="μόνο N ανά σετ (δοκιμή, δεν σώζει)")
    ap.add_argument("--only", default="", help=f"μόνο αυτά τα σετ, με κόμμα: {[k for k, _ in SETS]}")
    ap.add_argument("--fresh", action="store_true", help="αγνοεί το πάγωμα του Gemini")
    ap.add_argument("--rebuild", action="store_true", help="ξαναχτίζει το απομονωμένο store")
    args = ap.parse_args()
    if args.only and not set(args.only.split(",")) <= {k for k, _ in SETS}:
        print(f"!! --only από {[k for k, _ in SETS]}")
        return 1
    if args.compare and not os.path.exists(args.compare):
        print(f"!! Δεν υπάρχει {args.compare}")
        return 1
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY στο περιβάλλον — σταματάω")
        return 1
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη άλλο scoreboard (ή έμεινε κλειδαριά από διακοπή: σβήσε το {LOCK}) "
              f"— σταματάω. Δύο ταυτόχρονα τρεξίματα χαλάνε το κοινό store (ΠΑΓΙΔΕΣ, 28/9).")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
