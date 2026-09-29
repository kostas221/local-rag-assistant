"""Περιθώριο του φύλακα με το ΣΗΜΕΡΙΝΟ σύστημα — στο απομονωμένο store.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (28/9/2026):
    Ο φύλακας -2.6 βαθμονομήθηκε 10/8: χειρότερη σωστή -2.08 (q027), καλύτερη άσχετη
    -3.12 (q048), κενό 1.04, το -2.6 στο μέσο. Στις 16-18/8 άλλαξε η ΜΕΤΑΦΡΑΣΗ
    (γλωσσάρι ανά αρχείο, χωρίς τα δύο γραμμένα με το χέρι παραδείγματα) και η
    βαθμονόμηση ΔΕΝ ξαναμετρήθηκε. Αφορμή: η q025 («Ποιες τιμές ηλεκτρικού ρεύματος
    ανά κιλοβατώρα…») μεταφράζεται σήμερα «What cost per kilowatt-hour…» — το γλωσσάρι
    έχει τη λέξη «cost» και καταπίνει το «electricity» — και βγαίνει -2.93, ΙΔΙΟ στην
    παραγωγή και στο απομονωμένο store: κομμένη, ενώ απαντιέται.

ΓΙΑΤΙ ΟΧΙ ΣΚΕΤΟ measure_gate_margin.py:
    (α) ψάχνει στο ευρετήριο της ΠΑΡΑΓΩΓΗΣ, όπου ξένα αρχεία (και άλλων χρηστών)
        αλλάζουν τις βαθμολογίες BM25 — 4 αγγλικές ερωτήσεις άλλαξαν γι' αυτό·
    (β) καλεί το optimize_query ΧΩΡΙΣ τους όρους του scope (γράφτηκε πριν το
        γλωσσάρι ανά αρχείο). Για ερωτήσεις εκτός cache θα μετρούσε ΑΛΛΟ σύστημα.
    Εδώ τρέχει το ΙΔΙΟ script αυτούσιο, με δύο αντικαταστάσεις:
        store         = απομονωμένο (418 cloud chunks, ίδιο κείμενο 1-προς-1 με την παραγωγή)
        optimize_query = με domain/όρους όπως τα βάζει το search_documents
                         (corpus_glossary.assemble πάνω στα in-scope chunks)

ΜΕΤΑΦΡΑΣΕΙΣ: οι 20 ελληνικές του κύριου σετ από το cache της παραγωγής (τρέξιμο
25/9, ίδιο γλωσσάρι). Οι 2 του multihop (q054, q059) μεταφράζονται ΜΙΑ φορά και
σώζονται στο runs/gate_margin_translations.json -> 2ο τρέξιμο = ίδιο αποτέλεσμα.
Το cache της παραγωγής ΔΕΝ γράφεται (open_store).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    χειρότερη σωστή = q025 γύρω στο -2.93, ΚΑΤΩ από τον φύλακα (1 κομμένη in-corpus)
    άσχετες όπως 10/8 (καλύτερη -3.12)
    κενό 1.04 -> περίπου 0.2

    docker compose exec backend python evaluation/gate_margin_isolated.py --csv evaluation/runs/gate_margin_now.csv

ΦΑΣΗ 1.1 — ΙΔΙΟ SCRIPT ΜΕ ΑΛΛΟΝ RERANKER (29/9/2026):
    docker compose exec -e RERANKER_MODEL=Alibaba-NLP/gte-reranker-modernbert-base backend \\
        python evaluation/gate_margin_isolated.py --csv evaluation/runs/gate_margin_gte.csv
    Προϋπόθεση: ai_core φορτώνει τον reranker με activation Identity (αλλιώς το gte δίνει sigmoid).
    Ίδιες μεταφράσεις (runs/gate_margin_translations.json) -> η ΜΟΝΗ μεταβλητή είναι ο reranker.
    Σύγκριση με runs/gate_margin_now.csv (MiniLM: 59/61, κενό −1.31, κόβει q025 −2.93 / q059 −4.43).
    Η κλίμακα του gte είναι ΣΤΕΝΟΤΕΡΗ (−1.09…3.85 στο bench, MiniLM −4.44…9.27) -> τα κενά σε
    logits ΔΕΝ συγκρίνονται απευθείας· συγκρίνεται ΠΟΣΕΣ από τις 61 χωρίζονται καθαρά.
    ΠΡΟΒΛΕΨΗ (γραμμένη πριν):
      καλύτερη άσχετη (ooc)    0.7-1.5   (q019 ήδη 0.70 στο bench· υποψήφια για max η q048, GDPR)
      χειρότερη σωστή          0.3-1.5
      χωρίζονται καθαρά        57-60/61  -> δεν περιμένω ξεκάθαρη βελτίωση έναντι του 59/61
      q025 / q059              50/50 αν ΕΣΤΩ ΜΙΑ περνάει πάνω από την καλύτερη ooc
    ΑΠΟΤΕΛΕΣΜΑ (runs/gate_margin_gte.csv):
                        χειρότερες σωστές          καλύτερες άσχετες        κενό   καλύτερο κατώφλι
      MiniLM-L-12   q059 −4.43 · q025 −2.93      q048 −3.12 · q019 −4.44   −1.31   60/61 (στο −4.43: 0.01 από το q019)
      gte-modernbert q059 +1.41 · q025 +1.68     q050 +1.51 · q048 +1.43   −0.10   60/61 στο +1.59 (0.08 / 0.09)
      Κενό ως ποσοστό της απόστασης διαμέσων (σωστές − άσχετες): MiniLM −13% · gte −5%.
      -> Η q025 ΣΩΖΕΤΑΙ (πάνω από όλες τις άσχετες)· η q059 ΟΧΙ (κάτω από q050 ΚΑΙ q048).
      Σωστό chunk στη θέση 1: 41/56 -> 46/56. Ανά ερώτηση: καλύτερα 10 · χειρότερα 4 · ίδια 42
      (πρόσημο 10-4: p≈0.18, ΔΕΝ αποδεικνύεται). multi_hop ΜΙΚΤΑ: καλύτερα q044/q054/q057/q061,
      χειρότερα q045 (4->11) / q052 (3->6) / q060 (1->6).
      Πρόβλεψη: καλύτερη άσχετη 0.7-1.5 ✗ (1.51, και ήταν η q050 όχι η q048) · χειρότερη σωστή
      0.3-1.5 ✓ · 57-60/61 ✓ (60) · q025/q059 «50/50» = μη ελέγξιμη, δεν μετράει.
      ΚΑΙ ΤΑ ΔΥΟ ΚΑΤΩΦΛΙΑ ΕΙΝΑΙ ΣΤΗΝ ΚΟΨΗ ΤΟΥ ΞΥΡΑΦΙΟΥ: η βελτίωση του φύλακα είναι μία ερώτηση.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E

import corpus_glossary

TRANS = os.path.join(E.HERE, "runs", "gate_margin_translations.json")


def main() -> int:
    ai_core, _near = E.open_store()
    import measure_gate_margin as M  # ίδιο module ai_core -> βλέπει το απομονωμένο store

    if os.path.exists(TRANS):
        with open(TRANS, encoding="utf-8") as f:
            ai_core._translation_cache.update(json.load(f))

    where = ai_core._build_where(M.GOLDEN_CORPUS, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    scope_domain, scope_terms = corpus_glossary.assemble(allowed, ai_core._get_bm25_index())
    n_terms = len(scope_terms.split(", ")) if scope_terms else 0
    print(f"scope: {len(allowed)} chunks · domain {scope_domain or '(κενό -> ' + ai_core._CORPUS_DOMAIN + ')'}"
          f" · {n_terms} όροι γλωσσαρίου")

    orig = ai_core.optimize_query

    async def optimize_like_production(query, domain=None, terms=None):
        return await orig(query,
                          domain=scope_domain if domain is None else domain,
                          terms=scope_terms if terms is None else terms)

    ai_core.optimize_query = optimize_like_production
    code = asyncio.run(M.main())

    questions = [t["question"] for t in M.load_sets(M.DEFAULT_SETS)]
    used = {q: ai_core._translation_cache[q] for q in questions if q in ai_core._translation_cache}
    with open(TRANS, "w", encoding="utf-8") as f:
        json.dump(used, f, ensure_ascii=False, indent=1)
    print(f"Μεταφράσεις που χρησιμοποιήθηκαν ({len(used)}): {TRANS}")
    return code


if __name__ == "__main__":
    sys.exit(main())
