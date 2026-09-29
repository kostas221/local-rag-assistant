"""Διακόπτης embedder ΜΟΝΟ για την αξιολόγηση (Φάση 1.2) — ΚΑΜΙΑ αλλαγή στον κώδικα της παραγωγής.

    docker compose exec -e EVAL_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B backend python evaluation/scoreboard.py ...

ΓΙΑΤΙ ΕΤΣΙ: στο ai_core ο embedder είναι γραμμένος σταθερά (bge-m3) και φτιάχνει διανύσματα σε ΔΥΟ
μόνο σημεία — το ingest (μέσω του embedding function του store: `ai_core.sentence_transformer_ef`)
και την ερώτηση (`ai_core._embed_query`). Και τα δύο διαβάζονται ως globals τη στιγμή της κλήσης,
άρα αρκεί να αντικατασταθούν ΜΕΤΑ το `import ai_core` και ΠΡΙΝ ανοίξει το store. Αν το μοντέλο
κρατηθεί, η αλλαγή στην παραγωγή γίνεται ΜΕΤΑ, ως ρύθμιση, με ΑΠΟ/ΣΕ.

ΠΑΓΙΔΕΣ ΠΟΥ ΚΛΕΙΝΟΝΤΑΙ ΕΔΩ:
  - Το Qwen3-Embedding-0.6B βγάζει ΚΑΙ ΑΥΤΟ 1024 διαστάσεις: ο έλεγχος διάστασης του cache ερωτήσεων
    της παραγωγής (vector_db/_query_embeddings.npz) ΔΕΝ θα έπιανε ανάμειξη. Το αρχικό _embed_query
    δεν καλείται ποτέ και το _save_query_emb_cache γίνεται no-op -> το αρχείο της παραγωγής ΔΕΝ γράφεται.
  - Άλλο μοντέλο = άλλος χώρος διανυσμάτων: ΞΕΧΩΡΙΣΤΑ stores (κατάληξη με το όνομα του μοντέλου).
    Τα ids ξαναβγαίνουν με άλλο τυχαίο uuid στο τέλος, αλλά το πρόθεμα (αρχείο_σελίδα_κομμάτι) είναι
    μοναδικό -> το tie-break του RRF και η σειρά του γλωσσαρίου ΔΕΝ αλλάζουν (ίδιες μεταφράσεις).
  - Οδηγία ερώτησης: όπου το μοντέλο ορίζει prompt "query" (Qwen3) μπαίνει ΜΟΝΟ στην ερώτηση — τα
    κομμάτια χωρίς οδηγία, όπως ορίζει το ίδιο το μοντέλο. Μετρημένο (bench_embedder.py): με οδηγία
    84.7% κάλυψη dense top-30, χωρίς 81.9%.
  - float32: στο CPU το bf16 των βαρών του Qwen3 είναι αργό/ανακριβές.
"""
import os

import numpy as np

ENV = "EVAL_EMBED_MODEL"


class DocEmbeddingFunction:
    """Ίδια σύμβαση με το SentenceTransformerEmbeddingFunction της Chroma 0.4.6 (κείμενα -> λίστες,
    ΧΩΡΙΣ κανονικοποίηση: το ai_core κανονικοποιεί μόνο του στο _get_dense_matrix)."""

    def __init__(self, model):
        self._model = model

    def __call__(self, texts):
        return self._model.encode(list(texts), convert_to_numpy=True,
                                  normalize_embeddings=False).tolist()


def install(ai_core) -> str | None:
    """Αν ορίζεται EVAL_EMBED_MODEL (και διαφέρει από τον σημερινό), αντικαθιστά τον embedder.
    Επιστρέφει την κατάληξη για τα stores (π.χ. 'qwen3-embedding-0.6b') ή None αν δεν άλλαξε τίποτα."""
    name = os.getenv(ENV, "").strip()
    if not name or name == ai_core.EMBED_MODEL_NAME:
        return None
    if getattr(ai_core, "USE_BGE_SPARSE", False):
        raise RuntimeError("USE_BGE_SPARSE διαβάζει το sparse head του bge-m3 — ασύμβατο με άλλο embedder")
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"Διακόπτης embedder: φόρτωση {name} (float32) ...", flush=True)
    model = SentenceTransformer(name, device=ai_core.DEVICE, model_kwargs={"dtype": torch.float32})
    prompt = "query" if "query" in (model.prompts or {}) else None
    cache: dict[str, np.ndarray] = {}

    def embed_query(query: str) -> np.ndarray:
        v = cache.get(query)
        if v is None:
            v = np.asarray(model.encode([query], prompt_name=prompt, normalize_embeddings=True)[0],
                           dtype=np.float32)
            cache[query] = v
        return v

    ai_core.sentence_transformer_ef = DocEmbeddingFunction(model)
    ai_core._embed_query = embed_query
    ai_core._save_query_emb_cache = lambda: None
    ai_core.EMBED_MODEL_NAME = name
    suffix = name.split("/")[-1].lower()
    print(f"  ενεργός: {name} · οδηγία ερώτησης: {prompt!r} · stores με κατάληξη _{suffix}")
    return suffix
