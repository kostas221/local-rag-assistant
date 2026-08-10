"""Πόσα από τα chunks μας ΠΕΡΙΚΟΠΤΟΝΤΑΙ σιωπηλά από τον reranker;

ΤΟ ΕΡΩΤΗΜΑ:
Το `CrossEncoder(RERANKER_MODEL)` στο ai_core.py:171 δεν ορίζει max_length, οπότε
κληρονομεί το `max_seq_length` του μοντέλου — 512 tokens για το MiniLM-L-12. Το
sentence-transformers περικόπτει ΣΙΩΠΗΛΑ: καμία εξαίρεση, κανένα warning. Αν ένα
chunk ξεπερνά το όριο, ο reranker βαθμολογεί ΜΟΝΟ την αρχή του και η ουρά είναι
σαν να μην υπάρχει — αλλά το chunk μπαίνει ολόκληρο στο prompt του Gemini.

ΓΙΑΤΙ ΕΧΕΙ ΣΗΜΑΣΙΑ ΤΩΡΑ:
Πριν σαρώσουμε chunk_size, πρέπει να ξέρουμε αν ο άξονας «προς τα πάνω» είναι
ήδη κλειστός. Αν το 1500 ΗΔΗ περικόπτεται, το 2000/3000 δεν μπορεί να βοηθήσει
τον reranker — και μια σάρωση που το αγνοεί θα «ανακαλύψει» ότι τα μεγάλα chunks
δεν αποδίδουν, αποδίδοντάς το σε λάθος αιτία (chunking αντί για truncation).

ΤΙ ΜΕΤΡΑΕΙ:
Το πραγματικό ζεύγος που βλέπει ο reranker είναι (query, chunk). Το budget του
chunk είναι 512 μείον το query μείον τα special tokens. Μετράμε με ΠΡΑΓΜΑΤΙΚΟ
μεταφρασμένο query (τα queries φτάνουν αγγλικά — βλ. translate-then-retrieve).

ΜΗΔΕΝ ΚΟΣΤΟΣ: δεν φορτώνει bge-m3 ούτε τον cross-encoder, μόνο τον tokenizer.

    docker compose exec backend python evaluation/measure_chunk_tokens.py
    docker compose exec backend python evaluation/measure_chunk_tokens.py --show 5
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, "/app")

MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
COLLECTION = "ai_research_docs"

# Τυπικά μεταφρασμένα queries από το golden set. Το μήκος του query τρώει από το
# budget του chunk, οπότε το ΜΕΓΑΛΥΤΕΡΟ query δίνει το χειρότερο σενάριο.
SAMPLE_QUERIES = [
    "What is serverless computing?",
    "What are the main obstacles to cloud computing adoption and how can they be overcome?",
    "How does the paper compare the cost of transferring data over the network versus "
    "shipping physical disks, and what conclusion does it draw about bandwidth bottlenecks?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=3,
                    help="δείξε τα N χειρότερα chunks")
    args = ap.parse_args()

    import chromadb
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    limit = tok.model_max_length
    if limit > 100000:          # κάποια configs δίνουν sentinel αντί για αριθμό
        limit = 512
    print(f"Μοντέλο   : {MODEL}")
    print(f"Όριο      : {limit} tokens (query + chunk + special)\n")

    client = chromadb.PersistentClient(path=os.getenv("VECTOR_DB_PATH", "./vector_db"))
    got = client.get_collection(name=COLLECTION).get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    if not docs:
        print("*** Η συλλογή είναι άδεια.")
        return 1

    # Κόστος του query: το ζεύγος κωδικοποιείται ως [CLS] q [SEP] d [SEP]
    q_cost = {q: len(tok.encode(q)) for q in SAMPLE_QUERIES}
    worst_q = max(q_cost, key=lambda q: q_cost[q])

    lens = [len(tok.encode(d or "", add_special_tokens=False)) for d in docs]
    chars = [len(d or "") for d in docs]

    print(f"{len(docs)} chunks | χαρακτήρες: διάμ. {statistics.median(chars):.0f} "
          f"max {max(chars)}")
    print(f"tokens ανά chunk: διάμ. {statistics.median(lens):.0f} "
          f"μέσος {statistics.mean(lens):.0f} max {max(lens)}")
    print(f"αναλογία χαρακτήρες/token: {statistics.mean(chars) / statistics.mean(lens):.2f}\n")

    print("ΠΕΡΙΚΟΠΗ ανά μήκος query")
    print("-" * 62)
    print(f"{'query tokens':>13}{'budget chunk':>14}{'κομμένα':>10}{'ποσοστό':>10}")
    for q in sorted(SAMPLE_QUERIES, key=lambda x: q_cost[x]):
        budget = limit - q_cost[q] - 1        # +1 για το τελικό [SEP]
        cut = sum(1 for n in lens if n > budget)
        print(f"{q_cost[q]:>13}{budget:>14}{cut:>10}{100 * cut / len(lens):>9.1f}%")

    budget = limit - q_cost[worst_q] - 1
    over = [(n, c, m) for n, c, m in zip(lens, chars, metas) if n > budget]
    print("-" * 62)
    if not over:
        print(f"\nΚΑΝΕΝΑ chunk δεν περικόπτεται (χειρότερο: {max(lens)} <= {budget}).")
        print("-> Ο reranker βλέπει ΟΛΟΚΛΗΡΟ κάθε chunk. Ο άξονας chunk_size είναι")
        print(f"   ανοιχτός μέχρι ~{int(budget * statistics.mean(chars) / statistics.mean(lens))} "
              f"χαρακτήρες· πάνω από εκεί αρχίζει σιωπηλή απώλεια.")
    else:
        lost = sum(n - budget for n, _c, _m in over)
        print(f"\n*** {len(over)}/{len(lens)} chunks ΠΕΡΙΚΟΠΤΟΝΤΑΙ "
              f"({100 * len(over) / len(lens):.1f}%)")
        print(f"*** Χαμένα tokens: {lost} ({100 * lost / sum(lens):.1f}% του corpus)")
        print("*** Ο reranker βαθμολογεί ΜΙΣΟ chunk σε αυτές τις περιπτώσεις.\n")
        for n, c, m in sorted(over, reverse=True)[:args.show]:
            m = m or {}
            print(f"    {m.get('file_name', '?')}:{m.get('page', '?')}  "
                  f"{n} tokens / {c} χαρ.  -> χάνονται {n - budget}")

    print("\nΣΗΜ: το budget είναι για ΑΓΓΛΙΚΟ query (translate-then-retrieve). Το")
    print("     ελληνικό κείμενο δίνει ~2× tokens ανά χαρακτήρα στον ίδιο tokenizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
