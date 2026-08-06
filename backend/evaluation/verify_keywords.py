"""Έλεγχος ποιότητας των keywords ενός golden set — ΠΡΙΝ τρέξει οποιοδήποτε eval.

Το MRR/nDCG του eval_engine είναι *keyword-proxy*: ένα αποτέλεσμα θεωρείται σχετικό
αν το keyword υπάρχει ως substring μέσα του (βλ. eval_engine.calculate_mrr).
Άρα η αξία της μέτρησης εξαρτάται ΕΞ ΟΛΟΚΛΗΡΟΥ από την ποιότητα των keywords:

    keyword που ΔΕΝ υπάρχει στο corpus  ->  MRR 0 για πάντα  (σπασμένη ερώτηση)
    keyword που υπάρχει παντού          ->  υψηλό MRR από τύχη (τυφλή ερώτηση)

ΜΟΝΑΔΑ ΜΕΤΡΗΣΗΣ = ΣΕΛΙΔΑ, ΟΧΙ CHUNK. Το search_documents τελειώνει με
_expand_to_pages (ai_core.py:314) και επιστρέφει έως MAX_PAGES ΟΛΟΚΛΗΡΕΣ σελίδες.
Το eval κάνει substring match πάνω σε αυτές. Οπότε η «διακριτικότητα» ενός keyword
πρέπει να μετρηθεί σε επίπεδο σελίδας — σε επίπεδο chunk υποεκτιμάται.

ΤΥΧΑΙΟ BASELINE: για κάθε keyword υπολογίζεται ΑΚΡΙΒΩΣ (αναλυτικά, όχι με
προσομοίωση) το αναμενόμενο MRR αν το σύστημα επέστρεφε MAX_PAGES σελίδες στην
τύχη. Αυτό είναι το πάτωμα με το οποίο συγκρίνεται το πραγματικό MRR: ένα σκορ
0.85 σε benchmark με τυχαίο πάτωμα 0.40 δεν είναι το ίδιο με 0.85 σε πάτωμα 0.05.

Το script ΔΕΝ κάνει import το ai_core: δεν χρειάζεται embeddings ούτε reranker,
μόνο ανάγνωση των κειμένων από τη ChromaDB. Τρέχει σε δευτερόλεπτα, μηδέν κόστος
API, read-only στη βάση.

Χρήση (μέσα στο backend container):
    python evaluation/verify_keywords.py [golden_set.jsonl]

Exit code: 0 = κανένα σφάλμα, 1 = βρέθηκαν σπασμένα keywords (χρήσιμο ως CI gate).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

# --- Ρυθμίσεις ---
COLLECTION_NAME = "ai_research_docs"
HERE = Path(__file__).parent
DEFAULT_GOLDEN = HERE / "golden_set_20.jsonl"

# ΠΡΕΠΕΙ να ταυτίζεται με το _expand_to_pages(max_pages=8) του ai_core.py:314.
# Αν αλλάξει εκεί, άλλαξέ το κι εδώ — αλλιώς το τυχαίο baseline γίνεται λάθος.
MAX_PAGES = 8

# Κατώφλια με βάση το ΤΥΧΑΙΟ MRR, όχι αυθαίρετο ποσοστό: αν ένα keyword παίρνει
# μόνο του 0.50 MRR χωρίς καμία ανάκτηση, δεν μετράει retrieval — μετράει τη
# συχνότητα της λέξης.
RANDOM_TOO_HIGH = 0.50
RANDOM_WEAK = 0.25


def load_golden_set(path: Path) -> list[dict]:
    """Διαβάζει το .jsonl. Το `id` είναι προαιρετικό — αν λείπει (golden_set_20)
    παράγεται από τη σειρά της γραμμής, ώστε το output να είναι αναφέρσιμο."""
    questions = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        d = json.loads(line)
        d.setdefault("id", f"q{i:03d}")
        questions.append(d)
    return questions


def load_pages() -> tuple[list[str], list[tuple], int]:
    """Φέρνει τα chunks από τη ChromaDB και τα ΣΥΝΕΝΩΝΕΙ σε σελίδες — την ίδια
    μονάδα που βλέπει το eval μετά το _expand_to_pages.

    Η σειρά των chunks μέσα στη σελίδα δεν έχει σημασία εδώ: ψάχνουμε substring
    σε όλο το κείμενο της σελίδας, κι αυτό είναι ανεξάρτητο σειράς.

    Μόνο ανάγνωση κειμένων/metadata — δεν καλείται ποτέ embedding function,
    άρα δεν φορτώνεται κανένα μοντέλο.
    """
    db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    # Το κλειδί σελίδας περιλαμβάνει doc_id ΕΠΙΤΗΔΕΣ, όπως το _expand_to_pages
    # (ai_core.py:322). Αν το παραλείψεις, δύο ingests του ΙΔΙΟΥ αρχείου πέφτουν
    # πάνω στα ίδια κλειδιά και τα διπλότυπα γίνονται αόρατα — ενώ το eval τα
    # βλέπει ως ξεχωριστές σελίδες. (Ακριβώς έτσι κρύφτηκε ένα διπλότυπο εδώ.)
    buckets = defaultdict(list)
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        key = (str(meta.get("file_name", "?")), _page_num(meta.get("page")),
               meta.get("doc_id"))
        buckets[key].append(doc)

    keys = sorted(buckets)
    texts = ["\n".join(buckets[k]).lower() for k in keys]
    return texts, keys, len(docs)


def _page_num(value) -> int:
    """Η σελίδα στα metadata μπορεί να είναι int ή str — κανονικοποίηση για sort."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def random_mrr(hits: int, total: int, k: int = MAX_PAGES) -> float:
    """Αναμενόμενο MRR αν επιστρέφαμε k σελίδες ΣΤΗΝ ΤΥΧΗ (χωρίς επανάθεση).

    Ακριβής υπολογισμός, όχι προσομοίωση: διατρέχουμε τις θέσεις 1..k και σε κάθε
    μία πολλαπλασιάζουμε την πιθανότητα «δεν βρέθηκε hit μέχρι εδώ» επί την
    πιθανότητα «η θέση αυτή είναι hit», επί τη συνεισφορά 1/rank.
    """
    if hits <= 0 or total <= 0:
        return 0.0
    expected = 0.0
    p_still_searching = 1.0
    for rank in range(1, min(k, total) + 1):
        remaining = total - (rank - 1)
        expected += p_still_searching * (hits / remaining) * (1.0 / rank)
        p_still_searching *= (remaining - hits) / remaining
        if p_still_searching <= 0.0:
            break
    return expected


def analyse(keyword: str, pages_lower: list[str], page_keys: list[tuple]) -> dict:
    """Μετράει σε πόσες ΣΕΛΙΔΕΣ εμφανίζεται το keyword.

    ΚΡΙΣΙΜΟ: ο κανόνας ταιριάσματος είναι ΑΚΡΙΒΩΣ ο ίδιος με το
    eval_engine.calculate_mrr (case-insensitive substring). Αν αποκλίνει, το
    εργαλείο σταματά να προβλέπει τι θα κάνει το eval.
    """
    kw_lower = keyword.lower()
    locations = [key for text, key in zip(pages_lower, page_keys)
                 if kw_lower in text]
    hits = len(locations)
    total = len(pages_lower)
    return {
        "hits": hits,
        "total": total,
        "pct": (hits / total * 100) if total else 0.0,
        "random": random_mrr(hits, total),
        "sample": min(locations) if locations else None,
    }


def verdict(stats: dict, is_out_of_corpus: bool) -> tuple[str, str]:
    """Επιστρέφει (ετικέτα, σοβαρότητα) όπου σοβαρότητα = error|warn|ok.

    Για τις out_of_corpus ερωτήσεις ο κανόνας ΑΝΤΙΣΤΡΕΦΕΤΑΙ: εκεί το μηδέν είναι
    το ζητούμενο (τεστάρουν το relevance gate). Αν βρεθούν hits, η ερώτηση δεν
    είναι πια out-of-corpus — τυπικά επειδή προστέθηκε νέο PDF στο corpus.
    """
    if is_out_of_corpus:
        if stats["hits"] == 0:
            return "OK — gate", "ok"
        return "ΔΕΝ ΕΙΝΑΙ OUT-OF-CORPUS", "error"

    if stats["hits"] == 0:
        return "ΑΓΝΩΣΤΟ", "error"
    if stats["random"] > RANDOM_TOO_HIGH:
        return "ΠΟΛΥ ΚΟΙΝΟ", "warn"
    if stats["random"] > RANDOM_WEAK:
        return "ΑΔΥΝΑΜΟ", "warn"
    return "OK", "ok"


def main() -> int:
    golden_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLDEN
    if not golden_path.exists():
        golden_path = HERE / golden_path.name
    if not golden_path.exists():
        print(f"ΣΦΑΛΜΑ: δεν βρέθηκε το golden set: {golden_path}")
        return 1

    questions = load_golden_set(golden_path)
    pages_lower, page_keys, n_chunks = load_pages()
    if not pages_lower:
        print(f"ΣΦΑΛΜΑ: η συλλογή '{COLLECTION_NAME}' είναι άδεια — "
              "ανέβασε πρώτα τα PDF.")
        return 1

    corpus_files = sorted({key[0] for key in page_keys})
    print(f"\nGolden set : {golden_path.name} — {len(questions)} ερωτήσεις")
    print(f"Corpus     : {len(pages_lower)} σελίδες ({n_chunks} chunks) · "
          f"{len(corpus_files)} αρχεία")
    for fn in corpus_files:
        n = sum(1 for key in page_keys if key[0] == fn)
        print(f"             · {fn}  ({n} σελ.)")
    print(f"Προσομοίωση: τυχαία επιστροφή {MAX_PAGES} σελίδων "
          f"(= _expand_to_pages max_pages)\n")

    kw_width = min(max((len(kw) for q in questions for kw in q["keywords"]),
                       default=10), 24)
    counters = {"ok": 0, "warn": 0, "error": 0}
    problem_questions = []
    incorpus_randoms = []

    for q in questions:
        is_ooc = q.get("category") == "out_of_corpus"
        print(f"[{q['id']}] {q.get('category', 'uncategorized')} — "
              f"\"{q['question'][:58]}\"")

        worst = "ok"
        q_randoms = []
        for kw in q["keywords"]:
            stats = analyse(kw, pages_lower, page_keys)
            label, severity = verdict(stats, is_ooc)
            counters[severity] += 1
            q_randoms.append(stats["random"])
            if severity == "error" or (severity == "warn" and worst == "ok"):
                worst = severity

            where = (f"{stats['sample'][0][-20:]} p.{stats['sample'][1]}"
                     if stats["sample"] else "—")
            print(f"  {kw:<{kw_width}} {stats['hits']:>3}/{stats['total']} σελ "
                  f"({stats['pct']:>4.1f}%)  τυχαίο {stats['random']:.2f}  "
                  f"{where:<26} {label}")

        if not is_ooc and q_randoms:
            q_random = sum(q_randoms) / len(q_randoms)
            incorpus_randoms.append(q_random)
            print(f"  {'':<{kw_width}} → τυχαίο MRR ερώτησης: {q_random:.3f}")
        if worst != "ok":
            problem_questions.append((q["id"], worst))
        print()

    total = sum(counters.values())
    print("=" * 78)
    print(f"ΣΥΝΟΨΗ: {total} keywords · {counters['ok']} OK · "
          f"{counters['warn']} προειδοποιήσεις · {counters['error']} ΣΦΑΛΜΑΤΑ")
    if problem_questions:
        errs = [i for i, s in problem_questions if s == "error"]
        warns = [i for i, s in problem_questions if s == "warn"]
        if errs:
            print(f"  Ερωτήσεις με σφάλμα        : {', '.join(errs)}")
        if warns:
            print(f"  Ερωτήσεις με προειδοποίηση : {', '.join(warns)}")

    if incorpus_randoms:
        baseline = sum(incorpus_randoms) / len(incorpus_randoms)
        print("-" * 78)
        print(f"ΤΥΧΑΙΟ BASELINE (in-corpus, {len(incorpus_randoms)} ερωτήσεις): "
              f"MRR = {baseline:.3f}")
        print(f"  Δηλαδή: ένα σύστημα που επιστρέφει {MAX_PAGES} σελίδες ΣΤΗΝ ΤΥΧΗ")
        print(f"  σκοράρει {baseline:.3f} σε αυτό το benchmark, χωρίς καμία ανάκτηση.")
    print("=" * 78)

    return 1 if counters["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
