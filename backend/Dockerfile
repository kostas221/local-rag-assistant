# ΣΤΑΔΙΟ 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# ΝΕΟ: Εγκατάσταση των C++ Compilers για να μπορέσει να χτιστεί το hnswlib
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Δημιουργία Virtual Environment (Ανεξάρτητο από τον root)
RUN python -m venv /opt/venv

# Ενεργοποίηση του venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ΣΤΑΔΙΟ 2: Runtime
# ... (Από εδώ και κάτω άφησε το αρχείο όπως ακριβώς το είχες) ...

# ΣΤΑΔΙΟ 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Αντιγράφουμε ΜΟΝΟ το έτοιμο Virtual Environment από το προηγούμενο στάδιο
COPY --from=builder /opt/venv /opt/venv

# Ενεργοποίηση του venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

# Τα ML models (embeddings + reranker) κατεβαίνουν εδώ. Το path ΠΡΕΠΕΙ να ταιριάζει
# με το named volume στο docker-compose, αλλιώς ξανακατεβαίνουν σε κάθε restart.
ENV HF_HOME=/home/appuser/.cache/huggingface

# Ασφάλεια: Δημιουργία non-root χρήστη (με δικαίωμα εγγραφής στο cache)
# Δημιουργούμε ΚΑΙ το /app/vector_db (appuser-owned) ώστε το φρέσκο named volume
# να κληρονομήσει σωστά δικαιώματα — αλλιώς γίνεται root-owned και ο non-root
# appuser δεν μπορεί να γράψει το chroma.sqlite3 ("unable to open database file").
RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface /app/vector_db \
    && chown -R appuser /app /home/appuser
USER appuser

EXPOSE 8000

# --workers 1: ΕΠΙΛΟΓΗ ΧΩΡΗΤΙΚΟΤΗΤΑΣ, ΟΧΙ ΠΛΕΟΝ ΟΡΘΟΤΗΤΑΣ.
# Παλιότερα ήταν υποχρεωτικό: τα caches ακυρώνονταν από in-process μετρητή, άρα
# ένα ingest σε έναν worker άφηνε τους άλλους με stale index, σιωπηλά. Αυτό
# διορθώθηκε — η υπογραφή corpus περιλαμβάνει το mtime του store, ορατό σε κάθε
# διεργασία (βλ. ARCHITECTURE.md + tests/test_corpus_signature.py).
# ΓΙΑΤΙ ΠΑΡΑΜΕΝΕΙ 1, με νούμερα:
#   • ΜΝΗΜΗ: κάθε worker φορτώνει bge-m3 + cross-encoder = ~2.4GB. Δύο workers
#     διπλασιάζουν τη μνήμη για μηδέν επιπλέον throughput.
#   • CPU: μετρημένο (evaluation/concurrency_benchmark.py) ότι το throughput
#     κορεννύεται στους ~4 ταυτόχρονους χρήστες (~2.7 req/s) και ΠΕΦΤΕΙ στους 8.
#     Το φράγμα είναι οι 8 πυρήνες, όχι το πλήθος των workers· περισσότεροι
#     workers μοιράζονται την ίδια CPU και προσθέτουν context switching.
# Δηλαδή: το --workers 2 δεν είναι πια ΛΑΘΟΣ, απλώς δεν ωφελεί σε αυτό το
# μηχάνημα. Σε μεγαλύτερη CPU αξίζει επαναμέτρηση.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]