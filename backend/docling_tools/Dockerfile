# Απομονωμένο περιβάλλον ΜΟΝΟ για την εξαγωγή κειμένου (offline preprocessing).
#
# ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΟ IMAGE: το docling απαιτεί pydantic>=2, ενώ το backend είναι
# καρφωμένο σε pydantic 1.10 / fastapi 0.99.1 (αναπαραγωγιμότητα διπλωματικής).
# Τα δύο δεν συνυπάρχουν. Το ingest όμως ΔΕΝ είναι μέρος του API path — τρέχει
# μία φορά, offline, και αφήνει πίσω του καθαρό κείμενο σε αρχεία. Άρα το σωστό
# είναι να ζει σε δικό του container αντί να αναβαθμίσουμε όλο το serving stack.
FROM python:3.11-slim

# libgl/libglib: το docling φορτώνει opencv για την ανάλυση layout.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Χωρίς pin στην πρώτη δοκιμή — μόλις επιβεβαιωθεί ότι δουλεύει, καρφώνουμε
# την ακριβή έκδοση (pip freeze | grep docling) για αναπαραγωγιμότητα.
# Το pypdf μπαίνει κι εδώ γιατί το compare_extraction.py συγκρίνει τα δύο.
RUN pip install --no-cache-dir docling pypdf

# Τα μοντέλα layout/TableFormer κατεβαίνουν στο πρώτο τρέξιμο (~500MB) και
# μένουν στο named volume, ώστε να μην ξανακατεβαίνουν σε κάθε run.
ENV HF_HOME=/models

WORKDIR /work
