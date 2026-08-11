import asyncio
import json
import os
import re
import threading
import time
import unicodedata
import uuid

import chromadb
import numpy as np
import pymupdf
import torch
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google.api_core import exceptions as gexc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import gemini_rest
import metrics
from genai_compat import genai  # deprecated SDK, τεκμηριωμένο: βλ. genai_compat.py

# --- ΡΥΘΜΙΣΗ ΑΣΦΑΛΕΙΑΣ ΚΑΙ API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("ΠΡΟΣΟΧΗ: Δεν βρέθηκε το GEMINI_API_KEY στο .env αρχείο!")
genai.configure(api_key=GEMINI_API_KEY)

# Μοντέλο μέσω env (default gemini-2.5-flash, 1M context window) -> αλλάζει
# χωρίς rebuild στο deploy, π.χ. GEMINI_MODEL=gemini-2.5-pro.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# GenerationConfig:
#  • temperature=0.1 -> σχεδόν ντετερμινιστικές, ΣΤΑΘΕΡΕΣ απαντήσεις: ίδια ερώτηση
#    ≈ ίδια απάντηση κάθε φορά (σε factual QA η συνέπεια > η ποικιλία). Επίσης
#    reproducible eval + υψηλή πιστότητα (anti-hallucination). Με 0.3 η ίδια
#    ερώτηση έδινε άλλοτε αναλυτική κι άλλοτε σύντομη απάντηση.
#  • max_output_tokens=4096 -> καπάκι κόστους ΧΩΡΙΣ να κόβονται οι μεγάλες "λίστα
#    όλων των X" απαντήσεις. ΟΧΙ χαμηλότερο: το 2.5-flash είναι thinking model και
#    τα thinking tokens μετράνε στο budget -> πολύ χαμηλό όριο = κενή/κομμένη απάντηση.
GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.1,
    max_output_tokens=4096,
)
model = genai.GenerativeModel(GEMINI_MODEL, generation_config=GENERATION_CONFIG)

# ΕΠΙΚΥΡΩΜΕΝΟ ΜΕ JUDGE RUN (15 ερωτήσεις: 5 enumeration, 5 reasoning, 4 multi_hop,
# 1 out_of_corpus· σύγκριση ΑΝΑ ΕΡΩΤΗΣΗ με το judge_v2.csv):
#   budget=0    q047 accuracy 5->3 ΚΑΙ faithfulness 5->2 · q044/q046 completeness
#               5->4. ΟΛΕΣ οι υποβαθμίσεις σε multi_hop (3 από 4) — το μοντέλο
#               μαντεύει τη σύνδεση ανάμεσα σε δύο έγγραφα αντί να τη χτίσει.
#   budget=512  όλες επανήλθαν σε 5.0. Μέσες μεταβολές: accuracy +0.067,
#               completeness/relevance/faithfulness 0.000 -> ΜΗΔΕΝ κόστος.
# Το 512 δεν στραγγαλίζει: με ανοιχτό όριο το μοντέλο κατανάλωνε ~1.420 thinking
# tokens, με όριο 512 χρησιμοποιεί ~435 — δηλαδή το πλαφόν κόβει τη σπατάλη, όχι
# τη σκέψη. Αποτέλεσμα: TTFT 3.90s -> 2.78s (1.40x) με ταυτόσημη ποιότητα.
# Το 0 είναι 2.73x αλλά κοστίζει το faithfulness στα multi_hop — δεν αξίζει.
_tb = os.getenv("THINKING_BUDGET", "512")
THINKING_BUDGET = int(_tb) if _tb.strip() != "" else None
# Διακόπτης επιστροφής: "0" -> ξαναγυρνά στο SDK (χωρίς thinking control).
USE_REST_GENERATION = os.getenv("USE_REST_GENERATION", "1") not in ("0", "false", "False")


async def _gemini_generate(prompt, stream=False, retries=5):
    """Κλήση Gemini με exponential backoff στα 429 (ResourceExhausted).
    Τα waits (2,4,8,16,32s) καλύπτουν ένα ολόκληρο per-minute rate-limit window."""
    for attempt in range(retries):
        try:
            return await model.generate_content_async(prompt, stream=stream)
        except gexc.ResourceExhausted:
            wait = min(32, 2 ** (attempt + 1))  # 2,4,8,16,32
            logger.warning(
                f"Gemini rate limit (429). Retry σε {wait}s... "
                f"[{attempt + 1}/{retries}]")
            await asyncio.sleep(wait)
    raise RuntimeError("Gemini: εξάντληση retries λόγω rate limit (429).")


def _safe_chunk_text(chunk) -> str:
    """Ασφαλής εξαγωγή κειμένου από streaming chunk του Gemini.

    Το gemini-2.5-flash είναι thinking model: κάποια chunks περιέχουν ΜΟΝΟ
    thinking parts (κανένα text part). Ο quick accessor `chunk.text` ΠΕΤΑΕΙ τότε
    'response.text requires a valid Part' -> το broad except παρακάτω ακύρωνε ΟΛΗ
    την απάντηση (το bug στο Q15 «cold start»). Εδώ μαζεύουμε μόνο τα text parts
    και επιστρέφουμε '' στα thinking-only chunks αντί να σκάσει το stream."""
    try:
        return chunk.text or ""
    except Exception:
        parts = []
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts)
# ---------------------------------

logger.info("---> Αρχικοποίηση της AI Βάσης Δεδομένων (ChromaDB)...")
db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
chroma_client = chromadb.PersistentClient(path=db_path)

# --- Embeddings: BAAI/bge-m3 (multilingual, dense 1024-dim) ---
# Το bge-m3 ΔΕΝ θέλει prefixes (τέλος τα "passage:"/"query:"). Η standard
# SentenceTransformerEmbeddingFunction της ChromaDB κάνει embed ΚΑΙ τα έγγραφα
# (στο collection.add) ΚΑΙ τις ερωτήσεις (στο collection.query με query_texts),
# οπότε δεν χρειάζεται πλέον ξεχωριστό instance του μοντέλου.
# Συσκευή inference: αν υπάρχει CUDA GPU, τα μοντέλα τρέχουν εκεί (10-50× ταχύτερα)·
# αλλιώς CPU. Auto-detect ώστε ο ΙΔΙΟΣ κώδικας να δουλεύει και στα δύο deploy targets.
# (Για GPU χρειάζεται ΕΠΙΠΛΕΟΝ: CUDA build του torch στο image + nvidia container runtime.)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"---> Inference device: {DEVICE}")

# --- CPU threads για τα μοντέλα (ισχύει μόνο όταν DEVICE == "cpu") ----------
# ΜΕΤΡΗΜΕΝΟ (evaluation/measure_threads.py, 15 ζεύγη, αυτός ο container):
#   1 thread 773ms · 2: 444ms · 4: 317ms (default του torch) · 6: 261ms · 8: 235ms
# Το PyTorch διαλέγει μόνο του 4, με heuristic πάνω στους ΦΥΣΙΚΟΥΣ πυρήνες· κάτω
# από WSL2/Docker βγαίνει συντηρητικό. Με 8 threads ο cross-encoder — το 96% του
# warm retrieval — τρέχει 1.35x γρηγορότερα. ΚΑΜΙΑ επίπτωση στην ποιότητα:
# αλλάζει μόνο ο παραλληλισμός του ΙΔΙΟΥ υπολογισμού, όχι το αποτέλεσμα.
# ΠΡΟΣΟΧΗ ΣΤΟΝ ΦΟΡΤΟ: μετράει threads ΑΝΑ predict(). Τα requests τρέχουν σε
# asyncio.to_thread, άρα N ταυτόχρονες ερωτήσεις ζητούν N x TORCH_THREADS
# πυρήνες και αρχίζει context switching. Single-user demo/eval -> όλοι οι
# πυρήνες· υπό πραγματικό concurrency -> TORCH_THREADS=4.
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))  # 0 = auto (όλοι οι πυρήνες)
if DEVICE == "cpu":
    if TORCH_THREADS <= 0:
        TORCH_THREADS = (len(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity")
                         else (os.cpu_count() or 1))
    torch.set_num_threads(TORCH_THREADS)
    logger.info(f"---> CPU threads: {TORCH_THREADS}")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device=DEVICE
)

collection = chroma_client.get_or_create_collection(
    name="ai_research_docs",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"},  # bge-m3 είναι φτιαγμένο για cosine (default της Chroma είναι L2)
)

logger.info("---> Φόρτωση του Reranker (Αξιολογητή)...")
# ΑΓΓΛΙΚΟΣ reranker, όχι πολυγλωσσικός. ΓΙΑΤΙ: το optimize_query() μεταφράζει τα
# ελληνικά σε αγγλικά ΠΡΙΝ την ανάκτηση, και το corpus είναι αγγλικό -> ο
# cross-encoder βλέπει ΠΑΝΤΑ (EN ερώτημα, EN chunk). Το bge-reranker-v2-m3 (568M,
# XLM-R-large) πλήρωνε cross-attention για 100+ γλώσσες που δεν χρησιμοποιούνται
# πουθενά στο pipeline — η πολυγλωσσικότητα χρειάζεται μόνο στο dense embedding
# (ένα index για όλες τις γλώσσες) και στη γέννηση.
# ΜΕΤΡΗΜΕΝΟ (_reranker_compare.py — ίδια 15 υποψήφια, 50 ερωτήσεις, αυτή η CPU):
# ΜΕΤΡΗΜΕΝΟ ΔΥΟ ΦΟΡΕΣ. Ο 1ος γύρος (compare_rerankers.py, golden_set_50) έδειξε
# ισοπαλία 22M vs 568M -> κρατήθηκε το μικρό. Ο 2ος γύρος (compare_rerankers_hard.py)
# το ξαναέλεγξε στο hard set, γιατί το golden_set_50 αποδείχθηκε ότι έχει
# survivorship bias. Με ΙΔΙΑ εγγύηση out_of_corpus 5/5, ανά μοντέλο:
#   ms-marco-MiniLM-L-6-v2    22M    434 ms   hard  7/16 · kw@ διάμεσος 3.0
#   ms-marco-MiniLM-L-12-v2   33M    706 ms   hard 10/16 · kw@ διάμεσος 2.0
#   bge-reranker-v2-m3       568M   ~15 s     hard 12/16 · kw@ διάμεσος 1.0
#   bge-reranker-base        278M      —      hard  3/16 ΚΑΙ κανονικές 6/10 (!)
# Το L-12 παίρνει σχεδόν όλο το όφελος του 568M με 1.5x παραμέτρους. Το 568M
# δίνει +1 πραγματική ΚΑΙ +1 ψευδαίσθηση — καθαρό μηδέν για 17x μέγεθος.
# Το bge-base ΔΕΝ είναι μικρότερο v2-m3: χάνει ακόμα και από το L-6, γιατί
# βαθμολογεί ψηλά τα out_of_corpus -> το κατώφλι που τα κόβει σκοτώνει τα σωστά.
# ΠΡΟΣΟΧΗ: δίνει ΩΜΑ LOGITS (~[-11,+11]), ΟΧΙ sigmoid -> βλ. MIN_RERANK_SCORE.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
#   bge-reranker-v2-m3       sigmoid [0,1]         in-min  0.069 / out-max  0.013
#   ms-marco-MiniLM-L-6-v2   ωμά logits ~[-11,+11] in-min -1.80  / out-max -2.69
#   ms-marco-MiniLM-L-12-v2  ωμά logits ~[-11,+11] in-min -2.08  / out-max -3.12
# (measure_gate_margin.py, 61 ερωτήσεις: golden_set_50 + golden_multihop_new)
# Το L-12 ΜΕΓΑΛΩΣΕ το κενό: 0.89 -> 1.04 logits. Οι σωστές ανέβηκαν (διάμεσος
# 4.15 -> 4.62), οι άσχετες κατέβηκαν. Ολόκληρο το [-3.12, -2.08] δίνει 61/61.
# Το -2.6 είναι το ΜΕΣΟ του: 0.52 περιθώριο και προς τις δύο κατευθύνσεις — το
# σημείο που μεγιστοποιεί το ΧΕΙΡΟΤΕΡΟ περιθώριο.
# ΓΙΑΤΙ ΟΧΙ ΠΙΟ ΑΣΥΜΜΕΤΡΟ ΥΠΕΡ ΤΗΣ ΑΡΝΗΣΗΣ (όπως ήταν το -2.0 με 0.69/0.20):
# ο corrective agent δίνει πλέον δεύτερη ευκαιρία σε ό,τι κόβεται άδικα, ενώ μια
# διαρροή out_of_corpus ΔΕΝ έχει δίχτυ. Οι δύο πλευρές δεν είναι πια συμμετρικές
# στις συνέπειες με τον ίδιο τρόπο -> δεν πάμε κάτω από το μέσο.
# ΠΡΟΣΟΧΗ: το -2.0 ΣΠΑΕΙ με το L-12 — θα έκοβε το q027 (-2.08).
# 2η γραμμή άμυνας = το system prompt ("αν δεν είναι στο κείμενο, πες το")
# -> faithfulness 5.00/5 στο golden_set_50 το επιβεβαιώνει.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-2.6"))

# --- Corrective retrieval (2ο pass, ΜΟΝΟ όταν το gate έχει ήδη κόψει) ---
# άμυνα αντί να τη βοηθά. Σάρωση κατωφλίου (πραγματικές/ψευδαισθήσεις):
#   L-6  @ gate -2.0:  -2.0 -> 5/2 · 0.0 -> 4/1 · +1.0 -> 4/0 · +1.5 -> 3/0
#   L-12 @ gate -2.6:  ΟΛΟ το [-2.6, +0.8] -> 2/0 · +1.0 -> 1/0
# Τα 5 out_of_corpus μένουν κομμένα σε ΟΛΑ τα κατώφλια, και στα δύο μοντέλα.
# Το +0.4 κρατά ΑΚΡΙΒΩΣ την ίδια σχετική αυστηρότητα: 3.0 logits πάνω από το
# gate, όπως το +1.0 ήταν πάνω από το -2.0. Η αναλογία ήταν υπόθεση· η σάρωση
# την επιβεβαίωσε ανεξάρτητα (το 0.4 είναι μέσα στο καθαρό εύρος).
#
# ΠΡΟΣΟΧΗ — ΕΓΙΝΕ ΠΙΟ ΑΔΥΝΑΜΟ, ΟΧΙ ΛΙΓΟΤΕΡΟ: με το L-12 το gate κόβει 6 αντί 10
# hard ερωτήσεις (λύνει μόνο του τις h001/h003/h006/h013), άρα η βάση βαθμονόμησης
# ΣΥΡΡΙΚΝΩΘΗΚΕ από n=10 σε n=6 και οι σωσμένες από 4 σε 2. Το κατώφλι στηρίζεται
# πλέον σε ΔΥΟ ερωτήσεις (h005, h015). Αν μεγαλώσει το golden set, ΞΑΝΑΜΕΤΡΑ.
ENABLE_CORRECTIVE = os.getenv("ENABLE_CORRECTIVE", "1") not in ("0", "false", "False")
CORRECTIVE_MIN_SCORE = float(os.getenv("CORRECTIVE_MIN_SCORE", "0.4"))

# Πόσα candidates (από το RRF) περνάνε στον reranker — ο ΑΚΡΙΒΟΣ βήμα στη CPU.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
# Μέγεθος batch του cross-encoder. Το default του sentence-transformers είναι 32,
# οπότε και τα 15 υποψήφια μπαίνουν σε ΕΝΑ batch και γίνονται όλα pad στο μήκος
# του ΜΑΚΡΥΤΕΡΟΥ — ο reranker υπολογίζει padding tokens. Το smart batching
# (ταξινόμηση κατά μήκος) του sentence-transformers δρα ΜΕΤΑΞΥ batches, άρα με
# ένα batch δεν κάνει τίποτα. ΜΕΤΡΗΜΕΝΟ (tune_reranker_runtime.py, 15 ερωτήσεις):
#   batch=32 (default) 414ms · batch=15 402ms · batch=8 369ms · batch=4 361ms
#   Pearson ρ = 1.0000 και 0 αλλαγές στο top-1 chunk σε ΟΛΑ τα μεγέθη.
# Ίδιος ακριβώς υπολογισμός, 1.15x γρηγορότερα.
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "4"))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "30"))  # και για τα δύο σκέλη (dense + BM25)
EXPAND_INPUT = int(os.getenv("EXPAND_INPUT", "12"))
# Πόσες ΟΛΟΚΛΗΡΕΣ σελίδες φτάνουν τελικά στο LLM. Αυτό ΔΕΝ είναι βάθος υποψηφίων —
# είναι ο όγκος του context. Με ~4.500 χαρακτήρες/σελίδα, το 8 σημαίνει ~36.000
# χαρακτήρες (~9.000 tokens) ανά ερώτηση. Η βιβλιογραφία συνιστά πολύ μικρότερο
# top-k (3-5 κομμάτια) και τεκμηριώνει «lost in the middle» -25..-45% ανάκληση σε
# μακρύ context. ΜΕΤΡΗΜΕΝΟ (golden_set_50, καθαρό κείμενο, ντετερμινιστικά):
#   5 -> MRR 0.794 / cov 91.9%   ·   8 -> 0.799 / 94.8%   ·   12 -> 0.800 / 95.6%
# Το nDCG είναι ΤΑΥΤΟΣΗΜΟ (0.808) στα τρία. Το 12 δίνει +0.001 MRR για +50%
# context -> δεν αξίζει το latency ούτε το ρίσκο «lost in the middle». Μένει 8.
MAX_PAGES = int(os.getenv("MAX_PAGES", "8"))
# Πόσα από τα κορυφαία chunks μιας σελίδας αθροίζονται για να δώσουν το σκορ της.
# K=1 -> "max-P", δηλαδή ΑΚΡΙΒΩΣ η προηγούμενη συμπεριφορά (control).
# ΠΡΟΣΟΧΗ — Η ΑΡΧΙΚΗ ΔΙΚΑΙΟΛΟΓΗΣΗ ΔΕΝ ΙΣΧΥΕΙ ΠΛΕΟΝ. Το άθροισμα ήταν ασφαλές
# όσο ο reranker έβγαζε sigmoid στο [0,1] (bge-reranker-v2-m3: min 0.066, καμία
# αρνητική τιμή), όπου ένα αδύναμο 2ο chunk μπορούσε μόνο να προσθέσει λιγότερο.
# Το ms-marco-MiniLM βγάζει ΩΜΑ LOGITS με αρνητικές τιμές (~[-11,+11]) -> με K>1
# ένα αδύναμο chunk ΑΦΑΙΡΕΙ από το σκορ της σελίδας και μπορεί να τη ρίξει κάτω
# από μια σελίδα με ένα μόνο καλό chunk. Ο διακόπτης είναι στο 1, οπότε σήμερα
# δεν τρέχει τίποτα λάθος — αλλά K>1 θέλει ΠΡΩΤΑ επαναμέτρηση.
PAGE_SCORE_TOP_K = int(os.getenv("PAGE_SCORE_TOP_K", "1"))
# Ανώτατο πλήθος σελίδων από το ΙΔΙΟ αρχείο. Ίσο (ή μεγαλύτερο) από MAX_PAGES
# -> κανένα πρακτικό όριο, δηλαδή η προηγούμενη συμπεριφορά.
MAX_PAGES_PER_FILE = int(os.getenv("MAX_PAGES_PER_FILE", str(MAX_PAGES)))
# ΜΕΤΡΗΘΗΚΕ ΚΑΙ ΑΠΟΡΡΙΦΘΗΚΕ (golden_set_50, ντετερμινιστικά):
#   control (χωρίς)          in-corpus MRR 0.799 · direct_fact 0.900 · cov 94.8%
#   ως 3ο σκέλος ισότιμο     0.779 · 0.866 · 94.8%
#   ως 3ο σκέλος [1,.5,.5]   0.791 · 0.866 · 95.6%
# Βρίσκει ΠΕΡΙΣΣΟΤΕΡΑ σωστά (coverage +0.8pp, enumeration +0.012) αλλά τα
# ΚΑΤΑΤΑΣΣΕΙ χειρότερα: κάθε επιπλέον σκέλος αραιώνει το dense, που είναι το
# ισχυρότερο. Ο κώδικας μένει — ο διακόπτης κλειστός.
USE_BGE_SPARSE = os.getenv("USE_BGE_SPARSE", "0") not in ("0", "false", "False")
SPARSE_CACHE_PATH = os.path.join(db_path, "_sparse_weights.json")


def delete_file_from_db(filename: str, user_id: int | None = None,
                        doc_id: int | None = None):
    """Διαγράφει τα chunks ενός αρχείου. Αν δοθεί user_id, διαγράφει ΜΟΝΟ τα
    chunks αυτού του χρήστη — ώστε δύο χρήστες με ομώνυμο αρχείο να μην
    σβήνουν ο ένας τα δεδομένα του άλλου."""
    try:
        if doc_id is not None:
            where = {"doc_id": doc_id}
        elif user_id is not None:
            where = {"$and": [{"file_name": filename}, {"user_id": user_id}]}
        else:
            where = {"file_name": filename}
        collection.delete(where=where)
        _bump_corpus_version()  # invalidate το BM25 cache (λιγότερα chunks)
        logger.success(
            f"---> Επιτυχία: Τα δεδομένα του '{filename}' διαγράφηκαν.")
    except Exception as e:
        # ΞΑΝΑΠΕΤΑΜΕ: χωρίς αυτό ο caller (main.py delete_document) νόμιζε ότι η
        # διαγραφή πέτυχε και έσβηνε το Postgres row -> ορφανά chunks, ενεργά στο
        # retrieval αλλά αόρατα στο UI. Ίδιο failure mode με το race condition
        # ingest/delete που εξηγούσε την ασυμφωνία "386 vs 193 chunks" του v1.
        logger.error(f"---> Σφάλμα κατά τη διαγραφή από τη Vector DB: {e}")
        raise


def el_tokenize(text: str) -> list:
    """Tokenizer ανθεκτικός στα ελληνικά: lowercase + αφαίρεση τόνων/διακριτικών.
    Έτσι 'Ελληνικά' == 'ελληνικα' για το BM25 (οι χρήστες γράφουν συχνά άτονα)."""
    text = text.lower()
    # NFD: σπάει το γράμμα από τον τόνο -> πετάμε τα combining marks (κατηγορία Mn)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _has_greek(text: str) -> bool:
    """True αν το κείμενο περιέχει ελληνικούς χαρακτήρες (Greek + Greek Extended)."""
    return any(0x0370 <= ord(ch) <= 0x03FF or 0x1F00 <= ord(ch) <= 0x1FFF
               for ch in text)


# Απλό cache μεταφράσεων: η ίδια ερώτηση δεν ξαναμεταφράζεται (γλιτώνει API κλήσεις).
# Cache μεταφράσεων, ΜΟΝΙΜΟ στον δίσκο. Ζει μέσα στο volume της ChromaDB, οπότε
# επιβιώνει restarts και rebuilds (δεν γράφεται στο repo).
#
# ΓΙΑΤΙ ΕΧΕΙ ΣΗΜΑΣΙΑ ΠΕΡΑ ΑΠΟ ΤΙΣ ΚΛΗΣΕΙΣ API: το Gemini δεν είναι ντετερμινιστικό.
# Η ίδια ελληνική ερώτηση μεταφράστηκε σε δύο runs ως 'cloud computing 10 obstacles
# paper' και '10 main obstacles cloud computing' — διαφορετικό query, διαφορετικά
# embeddings, διαφορετικό MRR. Μετρήθηκε διακύμανση ~0.025 στο in-corpus MRR μεταξύ
# runs της ΙΔΙΑΣ διαμόρφωσης, δηλαδή θόρυβος μεγαλύτερος από πολλές πραγματικές
# βελτιώσεις. Με μόνιμο cache, τα eval runs γίνονται συγκρίσιμα μεταξύ τους.
_translation_cache_path = os.path.join(db_path, "_translation_cache.json")
_translation_lock = threading.Lock()


def _load_translation_cache() -> dict:
    try:
        with open(_translation_cache_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_translation_cache = _load_translation_cache()
logger.info(f"---> Translation cache: {len(_translation_cache)} εγγραφές "
            f"από {_translation_cache_path}")


def _save_translation_cache() -> None:
    """Γράψιμο μέσω προσωρινού αρχείου + os.replace: ατομικό, ώστε ένα crash
    στη μέση να μην αφήσει κατεστραμμένο JSON."""
    try:
        with _translation_lock:
            tmp = _translation_cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_translation_cache, f, ensure_ascii=False, indent=1)
            os.replace(tmp, _translation_cache_path)
    except OSError as e:
        logger.warning(f"--- Δεν αποθηκεύτηκε το translation cache: {e} ---")
# --- Cache των query embeddings -----------------------------------------------
# ΓΙΑΤΙ: μετρήθηκε ότι το embedding ΜΙΑΣ ερώτησης με το bge-m3 (568M) κοστίζει
# ~1190ms σε CPU — το 60% όλου του retrieval — ενώ το matmul των 418 διανυσμάτων
# είναι 0.5ms. Είναι το ακριβότερο βήμα του pipeline και είναι ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ:
# ίδιο κείμενο -> ίδιο διάνυσμα. Άρα cache-άρεται, όπως οι μεταφράσεις.
#
# ΔΕΣΜΕΥΜΕΝΟ ΣΤΟ ΜΟΝΤΕΛΟ: αν αλλάξει το embedding model, τα αποθηκευμένα
# διανύσματα δείχνουν σε άλλο χώρο (ίσως άλλη διάσταση) και θα έδιναν σιωπηλά
# λάθος αποτελέσματα -> το όνομα γράφεται μέσα στο αρχείο και ελέγχεται.
EMBED_MODEL_NAME = "BAAI/bge-m3"
# Διάσταση του bge-m3. Επικυρώνεται σε ΚΑΘΕ ανάγνωση και γραφή του cache: ένα
# διάνυσμα λάθος διάστασης σκάει στο matmul με αόριστο μήνυμα. Συνέβη — τα tests
# κάνουν monkeypatch τον embedder σε 8 διαστάσεις και μόλυναν το παραγωγικό npz.
EMBED_DIM = 1024
QUERY_EMB_CACHE_PATH = os.path.join(db_path, "_query_embeddings.npz")
# Cap αντί για LRU: οι ερωτήσεις είναι λίγες και επαναλαμβανόμενες. Το αρχείο
# ξαναγράφεται ολόκληρο σε κάθε νέα εγγραφή (~4KB/εγγραφή), άρα στο cap είναι
# ~100ms — αμελητέο απέναντι στα 1190ms που γλιτώνει το cold path.
QUERY_EMB_CACHE_MAX = int(os.getenv("QUERY_EMB_CACHE_MAX", "1000"))

_qemb_lock = threading.Lock()


def _load_query_emb_cache() -> dict:
    """{query: κανονικοποιημένο np.float32}. npz και όχι JSON: 1024 floats ανά
    εγγραφή είναι ~4KB σε binary έναντι ~20KB σε JSON, χωρίς rounding."""
    try:
        with np.load(QUERY_EMB_CACHE_PATH, allow_pickle=False) as z:
            if str(z["model"]) != EMBED_MODEL_NAME:
                logger.warning(
                    f"--- Query embedding cache από άλλο μοντέλο "
                    f"('{z['model']}' != '{EMBED_MODEL_NAME}') -> αγνοείται ---")
                return {}
            vecs = z["vectors"]
            if vecs.ndim != 2 or vecs.shape[1] != EMBED_DIM:
                logger.warning(
                    f"--- Query embedding cache με λάθος διάσταση "
                    f"({vecs.shape} != (n, {EMBED_DIM})) -> αγνοείται. "
                    f"Συνήθης αιτία: το έγραψε test με stubbed embedder. ---")
                return {}
            return dict(zip(z["queries"].tolist(), vecs))
    except (OSError, ValueError, KeyError):
        return {}


_query_emb_cache = _load_query_emb_cache()
logger.info(f"---> Query embedding cache: {len(_query_emb_cache)} εγγραφές "
            f"από {QUERY_EMB_CACHE_PATH}")


def _save_query_emb_cache() -> None:
    """Ατομικό γράψιμο (tmp + os.replace), ίδια σύμβαση με το translation cache.
    Το savez γράφεται σε file object ώστε να ΜΗΝ προσθέσει μόνο του '.npz'."""
    try:
        with _qemb_lock:
            items = list(_query_emb_cache.items())
            if not items:
                return
            tmp = QUERY_EMB_CACHE_PATH + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, model=np.array(EMBED_MODEL_NAME),
                    queries=np.array([q for q, _ in items]),
                    vectors=np.stack([v for _, v in items]))
            os.replace(tmp, QUERY_EMB_CACHE_PATH)
    except (OSError, ValueError) as e:
        logger.warning(f"--- Δεν αποθηκεύτηκε το query embedding cache: {e} ---")


def _embed_query(query: str) -> np.ndarray:
    """Κανονικοποιημένο embedding της ερώτησης, με μόνιμο cache.
    Το ΑΚΡΙΒΟΤΕΡΟ βήμα του pipeline. CPU-bound -> κάλεσέ το σε to_thread."""
    cached = _query_emb_cache.get(query)
    if cached is not None:
        return cached
    v = np.asarray(sentence_transformer_ef([query])[0], dtype=np.float32)
    nrm = np.linalg.norm(v)
    if nrm:
        v = v / nrm
        # Μην αποθηκεύεις ό,τι δεν είναι έγκυρο διάνυσμα του μοντέλου.
    if v.shape == (EMBED_DIM,) and len(_query_emb_cache) < QUERY_EMB_CACHE_MAX:
        _query_emb_cache[query] = v
        _save_query_emb_cache()
    return v

async def optimize_query(query: str):
    """Translate-then-Retrieve: τα έγγραφα είναι ΑΓΓΛΙΚΑ, οπότε αν η ερώτηση
    είναι ελληνική τη μεταφράζουμε σε αγγλικά ΜΟΝΟ για την ανάκτηση (retrieval).
    Η απάντηση μένει στη γλώσσα του χρήστη (το αναλαμβάνει το system prompt)."""
    # Αγγλική ερώτηση -> καμία κλήση API, χρησιμοποιείται ως έχει.
    if not _has_greek(query):
        return query

    if query in _translation_cache:
        return _translation_cache[query]

    try:
        # ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΛΥΝΕΙ (μετρημένο, measure_gate_margin.py):
        # χωρίς domain context ο μεταφραστής βγάζει καθημερινά αγγλικά, όχι όρους
        # του πεδίου: «μηχανήματα» -> "machinery" (βιομηχανικά!) αντί για "servers",
        # «ελέγχου» -> "control" αντί για "auditability". Το retrieval ψάχνει μετά
        # λέξεις που ΔΕΝ υπάρχουν στα papers.
        prompt = (
            "You are a translation assistant for an English-only academic search engine. "
            "The corpus is computer-science papers on cloud computing, serverless "
            "computing and distributed systems. Translate the user's question to English "
            "using the STANDARD TECHNICAL TERMINOLOGY of that field: map everyday words to "
            "the term a paper would actually use (e.g. Greek 'μηχανήματα' -> 'servers', "
            "NOT 'machinery'; 'έλεγχος' in a compliance context -> 'auditability'). "
            "Preserve every domain term already present in the question - never drop or "
            "generalize one. Output ONLY a concise English "
            "search query with the key terms. No quotes, no extra text.\n\n"
            f"User question: {query}"
        )
        
        english_query = (await gemini_rest.generate_once(
            prompt, model=GEMINI_MODEL, api_key=GEMINI_API_KEY)).strip(' "\'\n')
        _translation_cache[query] = english_query
        _save_translation_cache()
        logger.info(
            f"---> Retrieval translate: '{query[:40]}...' -> '{english_query}'")
        return english_query
    except Exception as e:
        logger.warning(f"--- Query translation failed: {e} ---")
        return query


async def _rewrite_query(question: str, history: list) -> str:
    """Conversational query rewriting: μετατρέπει follow-up ερώτηση σε ΑΥΤΟΝΟΜΟ
    search query χρησιμοποιώντας το ιστορικό. Π.χ. (συζήτηση για serverless) +
    «και το κόστος του;» -> «serverless billing cost». Καλείται ΜΟΝΟ όταν υπάρχει
    ιστορικό (follow-up) -> μηδέν επιπλέον κόστος στις πρώτες ερωτήσεις. Best-effort:
    σε σφάλμα/κενό επιστρέφει την αρχική ερώτηση. Χρησιμοποιείται ΜΟΝΟ για retrieval
    (η απάντηση & η γλώσσα ακολουθούν την αρχική ερώτηση)."""
    hist_text = "\n".join(
        f"{'USER' if m.get('role') == 'user' else 'ASSISTANT'}: {m.get('content', '')}"
        for m in history[-4:])
    prompt = (
        "Rewrite the user's follow-up question into a SINGLE self-contained search "
        "query for a document search engine, resolving pronouns/references using the "
        "conversation history. Keep the SAME language as the follow-up question. "
        "Output ONLY the rewritten query, no quotes, no extra text.\n\n"
        f"--- CONVERSATION HISTORY ---\n{hist_text}\n\n"
        f"Follow-up question: {question}\n\nStandalone search query:"
    )
    try:
        rewritten = (await gemini_rest.generate_once(
            prompt, model=GEMINI_MODEL, api_key=GEMINI_API_KEY)).strip(' "\'\n')
        if rewritten:
            logger.info(f"---> Query rewrite: '{question[:40]}...' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"--- Query rewrite failed: {e} ---")
    return question


_bm25_lock = threading.Lock()
_bm25_cache = {"version": None}
_corpus_version = 0

# --- Ταυτότητα corpus, ΟΡΑΤΗ ΜΕΤΑΞΥ ΔΙΕΡΓΑΣΙΩΝ ------------------------------
# ΤΟ ΠΡΟΒΛΗΜΑ: ο μετρητής _corpus_version ζει ΜΕΣΑ στη διεργασία. Με --workers>1
# ένα ingest στον worker A δεν τον αυξάνει στον worker B, που συνέχιζε να
# σερβίρει παλιό index ΣΙΩΠΗΛΑ, χωρίς κανένα σφάλμα — ο #1 τεκμηριωμένος
# περιορισμός του συστήματος.
# Η ΛΥΣΗ: στην υπογραφή μπαίνει και το mtime του ίδιου του store. Κάθε γράψιμο
# της ChromaDB το αλλάζει, ΟΠΟΙΑ διεργασία κι αν το έκανε.
# ΜΕΤΡΗΜΕΝΟ (evaluation/check_multiprocess_safety.py + probes):
#   • δεύτερη διεργασία γράφει -> η πρώτη το βλέπει ΧΩΡΙΣ restart (8/8 chunks).
#     Δουλεύει επειδή διαβάζουμε από το STORE, όχι από in-memory HNSW γράφημα:
#     η επιλογή που έγινε για ντετερμινισμό ξεκλειδώνει και το multi-worker.
#   • add ΚΑΙ delete αλλάζουν το mtime· 30 διαδοχικές ΑΝΑΓΝΩΣΕΙΣ ΟΧΙ — αλλιώς
#     το cache θα ξαναχτιζόταν σε κάθε ερώτηση.
#   • os.stat κοστίζει 2.4 microseconds -> 3 κλήσεις ανά ερώτηση = 0.002% των 450ms.
# Ο in-process μετρητής ΜΕΝΕΙ: τον αυξάνει ρητά το chunk_experiment.py, που
# αλλάζει collection χωρίς να γράψει στο ίδιο αρχείο.
_STORE_FILE = os.path.join(db_path, "chroma.sqlite3")


def _store_mtime() -> int:
    """mtime του store σε ns· 0 αν λείπει (πρώτο boot, πριν από κάθε ingest)."""
    try:
        return os.stat(_STORE_FILE).st_mtime_ns
    except OSError:
        return 0


def _corpus_signature():
    """(τοπικός μετρητής, mtime store). Αλλάζει είτε από ρητό bump μέσα στη
    διεργασία, είτε από γράψιμο οποιασδήποτε ΑΛΛΗΣ."""
    return (_corpus_version, _store_mtime())


def _bump_corpus_version():
    """Invalidate των caches μετά από ingest/delete (άλλαξε το corpus)."""
    global _corpus_version
    _corpus_version += 1


def _get_bm25_index():
    """Cached (bm25, ids, texts, metas, pos) πάνω σε ΟΛΟ το corpus. Ξαναχτίζεται
        μόνο όταν αλλάξει η υπογραφή του corpus. CPU-bound στο (επανα)χτίσιμο ->
    κάλεσέ το μέσα σε asyncio.to_thread."""
    sig = _corpus_signature()
    if _bm25_cache.get("version") == sig:
        return _bm25_cache
    with _bm25_lock:
        # double-check: άλλο thread μπορεί να το έχτισε όσο περιμέναμε το lock
        if _bm25_cache.get("version") == sig:
            return _bm25_cache
        data = collection.get()  # όλο το corpus (ids + documents + metadatas)
        ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
        tokenized = [el_tokenize(t) for t in texts]
        _bm25_cache.clear()
        _bm25_cache.update(
            version=sig, ids=ids, texts=texts, metas=metas,
            bm25=(BM25Okapi(tokenized) if tokenized else None),
            pos={id_: i for i, id_ in enumerate(ids)})
        logger.info(f"BM25 index (re)built: {len(ids)} chunks "
                    f"(version {_corpus_version}).")
        return _bm25_cache


def _bm25_sparse_ids(idx: dict, query: str, allowed_ids: list, top_n: int = 30):
    """BM25 scoring από το cached index, ΠΕΡΙΟΡΙΣΜΕΝΟ στα allowed_ids (authz).
    Επιστρέφει τα top_n ids κατά BM25. CPU-bound -> κάλεσέ το σε to_thread."""
    bm25 = idx["bm25"]
    if bm25 is None:
        return []
    scores = bm25.get_scores(el_tokenize(query))  # ευθυγραμμισμένο με idx["ids"]
    pos = idx["pos"]
    scored = [(scores[pos[i]], i) for i in allowed_ids if i in pos]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in scored[:top_n]]
# --- Exact dense αναζήτηση (αντικαθιστά το HNSW της Chroma) ------------------
_dense_lock = threading.Lock()
_dense_cache = {"version": None}


def _get_dense_matrix():
    """Cached (matrix, ids, pos) με ΚΑΝΟΝΙΚΟΠΟΙΗΜΕΝΑ embeddings όλου του corpus.
    Ίδιο pattern με το _get_bm25_index: ξαναχτίζεται μόνο σε αλλαγή corpus."""
    sig = _corpus_signature()
    if _dense_cache.get("version") == sig:
        return _dense_cache
    with _dense_lock:
        if _dense_cache.get("version") == sig:
            return _dense_cache
        data = collection.get(include=["embeddings"])
        M = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _dense_cache.clear()
        _dense_cache.update(
            version=sig, ids=data["ids"], matrix=M / norms,
            pos={id_: i for i, id_ in enumerate(data["ids"])})
        logger.info(f"Dense matrix (re)built: {M.shape[0]}x{M.shape[1]} "
                    f"(version {_corpus_version}).")
        return _dense_cache


def _dense_exact_ids(dm: dict, query: str, allowed_ids: list, top_n: int = 30):
    """Ακριβής (brute-force) cosine αντί για το approximate HNSW.

    ΓΙΑΤΙ: σε 422 chunks το matmul 422x1024 είναι ~0.5ms — το ANN δεν κερδίζει
    χρόνο, αλλά το in-memory γράφημα του HNSW ξαναχτίζεται από το WAL σε ΚΑΘΕ
    νέα διεργασία και μετρήθηκε να δίνει άλλο top-30 περίπου 1 στα 6 runs.
    Ισοδυναμία ελεγμένη: 30/30 overlap ΚΑΙ ίδια σειρά με το HNSW σε 3 ερωτήσεις.
    Μπόνους: διαβάζει τα διανύσματα από το store, όχι από το γράφημα — άρα το
    παλιό bug "77/422 vectors έλειπαν από το index" δεν θα μπορούσε να κρυφτεί.
    CPU-bound -> κάλεσέ το σε to_thread."""
    pos = dm["pos"]
    ids_allowed = [i for i in allowed_ids if i in pos]
    if not ids_allowed:
        return []
    v = _embed_query(query)          # cached: ~1190ms -> ~0 σε επανάληψη
    sims = dm["matrix"][[pos[i] for i in ids_allowed]] @ v
    # kind="stable": οι ισοβαθμίες λύνονται ΠΑΝΤΑ με τη σειρά των allowed_ids.
    order = np.argsort(-sims, kind="stable")[:top_n]
    return [ids_allowed[j] for j in order]
_sparse_lock = threading.Lock()
_sparse_cache = {"version": None}
_sparse_head = None


def _get_sparse_head():
    """(auto_model, tokenizer, linear_head) του ΗΔΗ φορτωμένου bge-m3.

    ΔΕΝ φορτώνει δεύτερο αντίγραφο του μοντέλου (θα ήταν +2.3GB RAM, όπως θα
    έκανε το FlagEmbedding): το sparse head είναι 1024+1 παράμετροι, ~4KB."""
    global _sparse_head
    if _sparse_head is None:
        from huggingface_hub import hf_hub_download
        transformer = sentence_transformer_ef._model[0]
        state = torch.load(hf_hub_download("BAAI/bge-m3", "sparse_linear.pt"),
                           map_location=DEVICE)
        head = torch.nn.Linear(transformer.auto_model.config.hidden_size, 1)
        head.load_state_dict(state)
        head.to(DEVICE).eval()
        _sparse_head = (transformer.auto_model, transformer.tokenizer, head)
    return _sparse_head


def _sparse_encode(texts: list, batch: int = 8) -> list:
    """[{token_id(str): βάρος}] ανά κείμενο, max ανά token. CPU-bound -> to_thread."""
    auto, tok, head = _get_sparse_head()
    special = set(tok.all_special_ids)
    out = []
    for i in range(0, len(texts), batch):
        feats = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            w = torch.relu(head(auto(**feats).last_hidden_state)).squeeze(-1)
        for ids, ws, mask in zip(feats["input_ids"], w, feats["attention_mask"]):
            d = {}
            for tid, val, m in zip(ids.tolist(), ws.tolist(), mask.tolist()):
                if not m or tid in special or val <= 0:
                    continue
                key = str(tid)
                if val > d.get(key, 0.0):
                    d[key] = round(val, 4)
            out.append(d)
    return out


def _get_sparse_index():
    """Cached sparse weights για ΟΛΟ το corpus, με persistence στο volume.

    Ο υπολογισμός είναι ~6 λεπτά για 418 chunks σε CPU — απαράδεκτο σε κάθε
    restart. Γι' αυτό γράφεται στον δίσκο και υπολογίζονται ΜΟΝΟ τα chunks που
    λείπουν (π.χ. μετά από ingest νέου αρχείου)."""
    sig = _corpus_signature()
    if _sparse_cache.get("version") == sig:
        return _sparse_cache
    with _sparse_lock:
        if _sparse_cache.get("version") == sig:
            return _sparse_cache
        idx = _get_bm25_index()          # ids + texts, ήδη cached
        stored = {}
        if os.path.exists(SPARSE_CACHE_PATH):
            try:
                with open(SPARSE_CACHE_PATH, encoding="utf-8") as f:
                    stored = json.load(f)
            except (ValueError, OSError) as e:
                logger.warning(f"Sparse cache μη αναγνώσιμο ({e}) -> ξαναχτίζεται.")
        missing = [i for i in idx["ids"] if i not in stored]
        if missing:
            logger.info(f"Sparse weights: υπολογίζονται {len(missing)} νέα chunks "
                        f"(από {len(idx['ids'])}).")
            texts = [idx["texts"][idx["pos"][i]] for i in missing]
            for cid, d in zip(missing, _sparse_encode(texts)):
                stored[cid] = d
            live = set(idx["ids"])
            stored = {k: v for k, v in stored.items() if k in live}  # σκούπισμα
            tmp = SPARSE_CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stored, f)
            os.replace(tmp, SPARSE_CACHE_PATH)  # ατομικό: ποτέ μισογραμμένο αρχείο
        _sparse_cache.clear()
        _sparse_cache.update(version=sig, weights=stored)
        logger.info(f"Sparse index έτοιμο: {len(stored)} chunks.")
        return _sparse_cache


def _bge_sparse_ids(sp: dict, query: str, allowed_ids: list, top_n: int = 30):
    """Top-n ids κατά BGE-M3 sparse ομοιότητα (dot product σε βάρη token).
    CPU-bound (forward pass της ερώτησης) -> κάλεσέ το σε to_thread."""
    weights = sp["weights"]
    qw = _sparse_encode([query])[0]
    if not qw:
        return []
    scored = []
    for cid in allowed_ids:
        d = weights.get(cid)
        if not d:
            continue
        s = sum(v * d[k] for k, v in qw.items() if k in d)
        if s > 0:
            scored.append((s, cid))
    # tie-break στο id: ντετερμινιστικό, ίδια λογική με το _rrf_fuse
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cid for _s, cid in scored[:top_n]]


def _rrf_fuse(dense_ids, sparse_ids, all_ids, all_texts, all_metadatas,
                k: int = 60, top_n: int = 15, extra_ids=None, weights=None,
                pos=None):
    """Reciprocal Rank Fusion: ενώνει 2-3 rankings σε ΕΝΑ score χρησιμοποιώντας
    ΜΟΝΟ τη θέση (rank) στη λίστα — άρα δεν χρειάζεται κοινή κλίμακα ανάμεσα σε
    cosine similarity, BM25 και sparse dot product. Καθαρή (pure) συνάρτηση ->
    δοκιμάζεται χωρίς ChromaDB/μοντέλα.

    extra_ids: προαιρετικό ΤΡΙΤΟ σκέλος (BGE-M3 sparse). Προαιρετικό ώστε τα
    υπάρχοντα tests και probes με δύο λίστες να δουλεύουν αμετάβλητα.

    k=60: ακαδημαϊκή σταθερά RRF. Doc που λείπει από μία λίστα παίρνει rank 1000
    (σχεδόν μηδενική συνεισφορά) αντί να αποκλείεται εντελώς.
    Επιστρέφει [(rrf_score, text, metadata)] φθίνουσα, κομμένο στα top_n."""
    lists = [dense_ids, sparse_ids] + ([extra_ids] if extra_ids else [])
    # weights: συνεισφορά κάθε σκέλους. None -> ισότιμα (κλασικό RRF). Χρειάζεται
    # όταν τα σκέλη δεν είναι ισοδύναμα: με δύο ΙΣΟΤΙΜΑ lexical σκέλη το lexical
    # ζυγίζει 2:1 έναντι του dense -> μετρήθηκε -0.034 στο direct_fact, κατηγορία
    # που δεν έχει καμία σχέση με το sparse.
    if weights is None:
        weights = [1.0] * len(lists)
    unique_ids = list(dict.fromkeys([i for lst in lists for i in lst]))
    rank_maps = [{id_: rank for rank, id_ in enumerate(lst)} for lst in lists]

        # Θέση ανά id ΜΙΑ φορά: το all_ids.index() μέσα στον βρόχο ήταν O(n²)
    # (50 υποψήφια x 418 ids). ΤΟ ΜΙΣΟ ΠΡΟΒΛΗΜΑ ΕΜΕΝΕ: και το ίδιο το χτίσιμο
    # του dict είναι O(N) σε ΚΑΘΕ ερώτηση. ΜΕΤΡΗΜΕΝΟ (scaling_benchmark.py):
    # 0.06ms στα 418 chunks · 3.9ms στα 50.000 · 22ms στα 200.000 — για δουλειά
    # που αφορά μόνο 30-45 υποψήφια. Ο caller έχει ΗΔΗ αυτό το dict cached
    # (idx["pos"] του BM25 index, πάνω στα ΙΔΙΑ ids)· περνώντας το, το κόστος
    # μηδενίζεται. Προαιρετικό ώστε tests και probes να καλούν με 5 ορίσματα.
    pos_by_id = pos if pos is not None else {
        id_: i for i, id_ in enumerate(all_ids)}
    fused = []
    for doc_id in unique_ids:
        rrf_score = sum(w / (k + rm.get(doc_id, 1000) + 1)
                        for w, rm in zip(weights, rank_maps))
        idx = pos_by_id[doc_id]
        fused.append((rrf_score, doc_id, all_texts[idx], all_metadatas[idx]))
    # Tie-break στο chunk id: χωρίς αυτό, chunks με ΙΔΙΟ rrf_score άλλαζαν σειρά
    # ανά διεργασία και το κόψιμο top_n έπαιρνε άλλο σύνολο υποψηφίων κάθε φορά.
    fused.sort(key=lambda x: (-x[0], x[1]))
    return [(s, t, m) for s, _id, t, m in fused[:top_n]]


def _chunk_idx_from_id(chunk_id: str) -> int:
    """Βγάζει το chunk index από το ID (μορφή '..._p{page}_c{idx}_{uuid}') ώστε
    να ανασυνθέσουμε τη σελίδα με τη ΣΩΣΤΗ σειρά. Χωρίς re-ingest."""
    m = re.search(r"_c(\d+)_", chunk_id)
    return int(m.group(1)) if m else 0


def _expand_to_pages(top_chunks, max_pages: int = 8):
    """Parent-document (page-level) retrieval: αντί για μεμονωμένα chunks,
    επιστρέφει ΟΛΟΚΛΗΡΕΣ τις σελίδες απ' όπου προέκυψαν τα top reranked chunks.
    Έτσι μια λίστα/πίνακας που απλώνεται σε μια σελίδα ανακτάται ΟΛΟΚΛΗΡΗ ->
    καλύτερη πληρότητα σε "λίστα όλων των X". CPU/IO -> κάλεσέ το σε to_thread.
    Authz: τα doc_id έχουν ήδη περάσει το where φίλτρο του χρήστη."""
    seen = []  # μοναδικές σελίδες, με σειρά relevance (σειρά των reranked chunks)
        # Σκορ σελίδας = άθροισμα των PAGE_SCORE_TOP_K καλύτερων chunks της.
    # Με K=1 ισοδυναμεί ΑΚΡΙΒΩΣ με την παλιά "σειρά πρώτης εμφάνισης": τα
    # top_chunks έρχονται ήδη φθίνουσα κατά rerank score, το dict κρατάει σειρά
    # εισαγωγής και το sorted() είναι stable -> ίδια σειρά, ίδιες ισοπαλίες.
    by_page = {}
    for score, _text, meta in top_chunks:
        key = (meta.get("doc_id"), meta.get("page"), meta.get("file_name"))
        by_page.setdefault(key, []).append(float(score))

    ranked = sorted(by_page.items(),
                    key=lambda kv: sum(sorted(kv[1], reverse=True)[:PAGE_SCORE_TOP_K]),
                    reverse=True)

    # Per-source cap: εμποδίζει ένα κυρίαρχο paper να καταλάβει όλες τις θέσεις,
    # ώστε να μείνει χώρος για δεύτερο έγγραφο (multi-hop).
    seen, per_file = [], {}
    for key, _scores in ranked:
        fname = key[2]
        if per_file.get(fname, 0) >= MAX_PAGES_PER_FILE:
            continue
        seen.append(key)
        per_file[fname] = per_file.get(fname, 0) + 1
        if len(seen) >= max_pages:
            break

    results = []
    for doc_id, page, file_name in seen:
        if doc_id is not None and doc_id != -1:
            where = {"$and": [{"doc_id": doc_id}, {"page": page}]}
        else:
            where = {"$and": [{"file_name": file_name}, {"page": page}]}
        pg = collection.get(where=where)
        if not pg["ids"]:
            continue
        ordered = sorted(zip(pg["ids"], pg["documents"]),
                         key=lambda p: _chunk_idx_from_id(p[0]))
        page_text = "\n".join(doc for _, doc in ordered)
        results.append((page_text, {"file_name": file_name, "page": page}))
    return results


def _build_where(target_filenames: list | None = None, user_id: int | None = None):
    """Φτιάχνει το ChromaDB where filter συνδυάζοντας:
    - δικαίωμα πρόσβασης (δικά μου έγγραφα ή public), αν δοθεί user_id
    - τα επιλεγμένα αρχεία (target_filenames), αν δοθούν
    """
    # ΣΗΜ: η chromadb 0.4.6 ΔΕΝ υποστηρίζει $in — χρησιμοποιούμε $or από $eq.
    clauses = []
    if user_id is not None:
        clauses.append({"$or": [{"user_id": user_id}, {"is_public": True}]})
    if target_filenames:
        if len(target_filenames) == 1:
            clauses.append({"file_name": target_filenames[0]})
        else:
            clauses.append(
                {"$or": [{"file_name": f} for f in target_filenames]})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# Το prompt είναι η ΠΛΟΥΣΙΑ εκδοχή (v1). Δοκιμάστηκε και λακωνική εκδοχή με
# ρητούς κανόνες «μία έννοια, χωρίς παραγέμισμα» (prompts/corrective_v2.txt):
# ΧΕΙΡΟΤΕΡΗ σε κάθε κατώφλι (3/2 έναντι 5/2), γιατί παρήγαγε queries 2-3 λέξεων
# που δίνουν στον reranker ελάχιστο σήμα. Χειρότερο ακόμα: στο v2 η ΥΨΗΛΟΤΕΡΗ
# βαθμολογία ολόκληρου του retry ήταν ψευδαίσθηση -> κανένα κατώφλι δεν τη
# φιλτράρει. Το v1 κρατά τις ψευδαισθήσεις στα ΧΑΜΗΛΑ σκορ, άρα φιλτράρονται.
_CORRECTIVE_PROMPT = (
    "The following search query returned no sufficiently relevant results from a "
    "corpus of computer-science papers on cloud computing, serverless computing "
    "and distributed systems.\n\n"
    "Rewrite it as a keyword-rich search query using the standard technical "
    "terminology of that field. Replace vague or conversational wording with the "
    "terms a paper would actually use. Do NOT add facts, system names or numbers "
    "that are not implied by the original question. If the question is about a "
    "topic OUTSIDE that field, return it unchanged.\n\n"
    "Output ONLY the rewritten query. No quotes, no extra text.\n\n"
    "Original query: {query}\nRewritten query:"
)


async def _corrective_retry(query: str, first_best: float, allowed_ids: list,
                            idx: dict, dm: dict):
    """2ο pass μετά από κοπή του gate. Επιστρέφει sorted_final, ή None αν πρέπει
    να παραμείνει κομμένο. Best-effort: ΚΑΘΕ σφάλμα -> None (δηλαδή η σημερινή
    συμπεριφορά), ποτέ εξαίρεση προς τον χρήστη."""
    if not ENABLE_CORRECTIVE:
        metrics.inc("rag_corrective_skipped_total", {"reason": "disabled"})
        return None
    metrics.inc("rag_corrective_attempts_total")
    try:
        new_query = (await gemini_rest.generate_once(
            _CORRECTIVE_PROMPT.format(query=query),
            model=GEMINI_MODEL, api_key=GEMINI_API_KEY)).strip(' "\'\n')
    except Exception as e:
        logger.warning(f"--- Corrective rewrite απέτυχε: {e} ---")
        metrics.inc("rag_corrective_skipped_total", {"reason": "rewrite_error"})
        return None
    # Ίδιο ερώτημα -> ίδιο αποτέλεσμα· μη σπαταλάς 450ms CPU για να το επιβεβαιώσεις.
    if not new_query or new_query.strip().lower() == query.strip().lower():
        logger.info("Corrective: το rewrite δεν άλλαξε το ερώτημα -> κομμένο")
        metrics.inc("rag_corrective_skipped_total", {"reason": "identical"})
        return None

    logger.info(f"---> Corrective retry: '{query[:50]}' -> '{new_query}'")
    dense_ids = await asyncio.to_thread(
        _dense_exact_ids, dm, new_query, allowed_ids,
        min(DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = await asyncio.to_thread(
        _bm25_sparse_ids, idx, new_query, allowed_ids, DENSE_CANDIDATES)
    rrf_sorted = _rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                           idx["metas"], k=60, top_n=RERANK_CANDIDATES,
                           pos=idx["pos"])
    pairs = [[new_query, item[1]] for item in rrf_sorted]
    cross_scores = await asyncio.to_thread(
        lambda: reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE))
    sorted_final = sorted(
        zip(cross_scores, [it[1] for it in rrf_sorted],
            [it[2] for it in rrf_sorted]), key=lambda x: x[0], reverse=True)

    if sorted_final[0][0] < CORRECTIVE_MIN_SCORE:
        logger.info(
            f"Corrective gate: best={sorted_final[0][0]:.2f} < "
            f"{CORRECTIVE_MIN_SCORE} -> παραμένει κομμένο")
        return None
    logger.info(
        f"Corrective retry ΠΕΤΥΧΕ: {first_best:.2f} -> {sorted_final[0][0]:.2f}")
    metrics.inc("rag_corrective_success_total")
    return sorted_final


async def search_documents(raw_query: str, target_filenames: list | None = None,
                           user_id: int | None = None):
    # Αν ο χρήστης δεν έχει επιλέξει κανένα (έγκυρο) αρχείο, μη γυρνάς τυχαία
    # αποτελέσματα — απάντησε "τίποτα".
    if target_filenames is not None and len(target_filenames) == 0:
        return []

    metrics.inc("rag_queries_total")

    # 0. Κρυφή μετάφραση/εμπλουτισμός (Gemini API)
    query = await optimize_query(raw_query)

    where_filter = _build_where(target_filenames, user_id)

    # 1. Allowed ids (AUTHORIZATION single-source στη Chroma). Παίρνουμε ΜΟΝΟ
    # ids/metas — ΟΧΙ το βαρύ documents· τα κείμενα τα έχει το cached BM25 index.
        # 1. Allowed ids (AUTHORIZATION single-source στη Chroma). include=[] ->
    # ΜΟΝΟ τα ids. Τα metadatas ζητούνταν για 418 chunks σε ΚΑΘΕ ερώτηση και
    # δεν διαβάζονταν ποτέ· τα κείμενα τα έχει ήδη το cached BM25 index.
    allowed = await asyncio.to_thread(
        lambda: collection.get(where=where_filter, include=[]))
    allowed_ids = allowed["ids"]
    if not allowed_ids:
        return []

    # Cached BM25 index πάνω σε ΟΛΟ το corpus (χτίζεται μία φορά ανά version).
    idx = await asyncio.to_thread(_get_bm25_index)
    all_ids, all_texts, all_metadatas = idx["ids"], idx["texts"], idx["metas"]

        # 2. Dense Search — bge-m3 (cosine), EXACT. Το authz γίνεται στα allowed_ids,
    # που έχουν ήδη περάσει το where φίλτρο. to_thread: CPU-bound embed + matmul.
    dm = await asyncio.to_thread(_get_dense_matrix)
    dense_ids = await asyncio.to_thread(
        _dense_exact_ids, dm, query, allowed_ids,
        min(DENSE_CANDIDATES, len(allowed_ids)))

    # 3. Sparse Search (BM25 από το cache), περιορισμένο στα allowed_ids
        # 3. Sparse Search (BM25 από το cache), περιορισμένο στα allowed_ids
    sparse_ids = await asyncio.to_thread(
        _bm25_sparse_ids, idx, query, allowed_ids, DENSE_CANDIDATES)

    # 3β. BGE-M3 sparse — ΤΡΙΤΟ σκέλος. Αποτυγχάνει σε ΑΛΛΕΣ ερωτήσεις απ' ό,τι
    # το BM25 (κερδίζει στα multi_hop, χάνει στα CLI flags), γι' αυτό προστίθεται
    # αντί να το αντικαθιστά. Το RRF συγχωνεύει χωρίς κοινή κλίμακα.
    bge_ids = []
    if USE_BGE_SPARSE:
        sp = await asyncio.to_thread(_get_sparse_index)
        bge_ids = await asyncio.to_thread(
            _bge_sparse_ids, sp, query, allowed_ids, DENSE_CANDIDATES)

    # 4. Reciprocal Rank Fusion (RRF): ένωση των σκελών -> top-15
        # Το συνολικό lexical βάρος μένει 1.0, μοιρασμένο στα δύο σκέλη, ώστε η
    # προσθήκη του BGE sparse να ΜΗΝ υποβαθμίζει το dense.
    rrf_sorted = _rrf_fuse(dense_ids, sparse_ids, all_ids, all_texts,
                           all_metadatas, k=60, top_n=RERANK_CANDIDATES,
                           extra_ids=bge_ids,
                           weights=[1.0, 0.5, 0.5] if bge_ids else None,
                           pos=idx["pos"])

    # 5. Reranking με τον Cross-Encoder — το βαρύτερο CPU κομμάτι, σε thread
    pairs = [[query, item[1]] for item in rrf_sorted]
    cross_scores = await asyncio.to_thread(
        lambda: reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE))

    final_combined = list(zip(cross_scores, [item[1] for item in rrf_sorted],
                              [item[2] for item in rrf_sorted]))
    sorted_final = sorted(final_combined, key=lambda x: x[0], reverse=True)

    # --- Relevance gate (anti-hallucination) — κατώφλι/σκεπτικό: MIN_RERANK_SCORE ---
    if sorted_final[0][0] < MIN_RERANK_SCORE:
        logger.info(
            f"Relevance gate: best={sorted_final[0][0]:.2f} < {MIN_RERANK_SCORE} "
            f"-> κανένα αρκετά σχετικό chunk")
        metrics.inc("rag_gate_blocked_total")
        # Corrective 2ο pass: ΜΟΝΟ εδώ, δηλαδή μόνο όταν το σύστημα ήδη απέτυχε.
        # Το happy path (61/61 ερωτήσεις του golden set) δεν το βλέπει ποτέ.
        sorted_final = await _corrective_retry(
            query, sorted_final[0][0], allowed_ids, idx, dm)
        if sorted_final is None:
            return []

    # Parent-document (page-level) expansion: επιστρέφουμε ΟΛΟΚΛΗΡΕΣ ΣΕΛΙΔΕΣ
    # (όχι μεμονωμένα chunks) από τα top reranked chunks -> καλύτερη πληρότητα
    # σε ερωτήσεις τύπου "λίστα όλων των X" (π.χ. όλα τα εμπόδια του cloud).
    return await asyncio.to_thread(_expand_to_pages, sorted_final[:EXPAND_INPUT],
                                   MAX_PAGES)

# --- ΜΕΤΑΤΡΟΠΗ ΣΕ ASYNC GENERATOR ---


# Στυλ απάντησης ανά persona (η γλώσσα & ο κανόνας μη-ψευδαίσθησης μένουν σταθερά)
PERSONA_STYLES = {
    "Researcher": "Maintain an objective, scientific tone. Provide a THOROUGH, well-structured answer that fully addresses the question using ALL relevant details from the sources. State the main finding first, then expand with supporting evidence, context and nuances. Use as many sentences as needed for completeness (typically 4-8); do NOT artificially shorten.",
    "Educator": "Explain clearly and simply, as if teaching a student, with a short clarifying example if helpful (3-5 sentences).",
    "Concise": "Be extremely concise: 1-2 sentences or short bullet points, only the essential finding.",
}


async def ask_ai(question, target_filenames, history=None, user_id=None, persona="Researcher",
                 precomputed=None):
    """precomputed: έτοιμο αποτέλεσμα ανάκτησης — χρησιμοποιείται ΜΟΝΟ από το eval
    ώστε να μη γίνει δεύτερη φορά η ίδια (ακριβή σε CPU) αναζήτηση. Η εφαρμογή δεν
    το περνά ποτέ, οπότε η συμπεριφορά της παραμένει αμετάβλητη."""
    if history is None:
        history = []

    # Conversational query rewriting: σε follow-up (υπάρχει ιστορικό) ξαναγράφουμε
    # την ερώτηση σε αυτόνομο query ΜΟΝΟ για το retrieval. Η ίδια η ερώτηση (question)
    # μένει αναλλοίωτη για τη γέννηση της απάντησης & τη γλώσσα.
    retrieval_query = await _rewrite_query(question, history) if history else question

    # --- MLOps: μέτρηση χρόνου φάσης ανάκτησης (retrieval) ---
    t_retrieval = time.perf_counter()
    top_3_data = (precomputed if precomputed is not None
                  else await search_documents(retrieval_query, target_filenames,
                                              user_id=user_id))
    retrieval_time = time.perf_counter() - t_retrieval
    metrics.observe("rag_retrieval_seconds", retrieval_time)

    if not top_3_data:
        metrics.inc("rag_answers_total", {"outcome": "no_context"})
        yield {"type": "sources", "data": []}
        msg = ("Δεν βρέθηκε απάντηση στα επιλεγμένα έγγραφα."
               if _has_greek(question)
               else "The selected documents do not contain an answer to this question.")
        yield {"type": "text", "data": msg}
        return
    context_text = ""
    sources_list = []

    for text, meta in top_3_data:
        file_n = meta.get('file_name', 'Unknown File')
        page_n = meta.get('page', '?')
        context_text += f"\n[Source: {file_n}, Page: {page_n}]\n{text}\n"
        # Πλούσιο αντικείμενο αντί για string: το UI δείχνει και απόσπασμα
        # του chunk -> ο χρήστης ΕΠΑΛΗΘΕΥΕΙ από πού βγήκε η απάντηση.
        sources_list.append({
            "file": file_n,
            "page": page_n,
            "preview": text[:400] + ("…" if len(text) > 400 else ""),
        })

    history_text = ""
    if history:
        history_text = "--- CONVERSATION HISTORY ---\n"
        for msg in history[-4:]:
            role = "USER" if msg.get("role") == "user" else "ASSISTANT"
            history_text += f"{role}: {msg.get('content')}\n\n"
        history_text += "---------------------------\n\n"

    # --- SYSTEM PROMPT: γλώσσα + persona-στυλ + κανόνας μη-ψευδαίσθησης ---
    persona_style = PERSONA_STYLES.get(persona, PERSONA_STYLES["Researcher"])
    system_prompt = f"""You are an expert Research Assistant. Provide precise, evidence-based answers using ONLY the provided SOURCE TEXT.

    STRICT PROTOCOLS:
    1. LANGUAGE ENFORCEMENT: You MUST answer in the EXACT SAME LANGUAGE as the user's question. Greek question -> Greek answer. English question -> English answer. When answering in Greek, write FLUENT, natural, grammatically correct Greek; keep established English technical terms (e.g., cloud, serverless, API, FaaS, BaaS) in English.
    2. STYLE: {persona_style}
    3. NO HALLUCINATIONS: Base your answer EXCLUSIVELY on the SOURCE TEXT. If the answer is not in the text, clearly state that you cannot find the answer in the provided documents (in the SAME language as the question).
    4. FORMATTING: If you use a numbered list, number the items sequentially starting from 1 (1, 2, 3, ...). NEVER reuse the numbering from the source document, and ensure any count you state (e.g. "five challenges") exactly matches the number of items you list.
    5. COMPLETENESS: Do NOT state a total count of items (e.g. "five obstacles") unless that exact number is explicitly written in the SOURCE TEXT. List EVERY relevant item present in the sources; if the sources appear partial, present what you found without claiming the list is complete.
    """

    # Ντετερμινιστική οδηγία γλώσσας: ανιχνεύουμε ΕΜΕΙΣ τη γλώσσα της τρέχουσας
    # ερώτησης (_has_greek) και τη ΕΠΙΒΑΛΛΟΥΜΕ ρητά -> δεν παρασύρεται το μοντέλο
    # από ελληνικό/αγγλικό ιστορικό (π.χ. αγγλική ερώτηση σε ελληνικό chat).
    lang_rule = (
        "ΑΠΑΝΤΗΣΕ ΑΠΟΚΛΕΙΣΤΙΚΑ ΣΤΑ ΕΛΛΗΝΙΚΑ — ανεξάρτητα από τη γλώσσα του ιστορικού."
        if _has_greek(question) else
        "ANSWER EXCLUSIVELY IN ENGLISH — regardless of the conversation history language."
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- SOURCE TEXT ---\n{context_text}\n\n"
        f"{history_text}"
        f"CURRENT QUESTION: {question}\n\n"
        f"REMINDER: {lang_rule}\n\n"
        f"ANSWER:"
    )

        # Dedup ανά (αρχείο, σελίδα) — τα dicts δεν μπαίνουν σε set().
    seen, unique_sources = set(), []
    for s in sources_list:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)
    yield {"type": "sources", "data": unique_sources}

    t_generation = time.perf_counter()
    prompt_tokens = out_tokens = thinking_tokens = None
    produced_text = False  # για fallback αν το μοντέλο γυρίσει ΜΟΝΟ thinking
    try:
        if USE_REST_GENERATION:
            # REST: το ΜΟΝΟ μονοπάτι που δέχεται thinkingConfig. Τα thinking
            # parts έρχονται σημαδεμένα με "thought": true και φιλτράρονται ρητά
            # μέσα στο gemini_rest — όχι με heuristic πάνω σε exception.
            async for kind, data in gemini_rest.stream_generate(
                    full_prompt, model=GEMINI_MODEL, api_key=GEMINI_API_KEY,
                    temperature=GENERATION_CONFIG.temperature,
                    max_output_tokens=GENERATION_CONFIG.max_output_tokens,
                    thinking_budget=THINKING_BUDGET):
                if kind == "text":
                    produced_text = True
                    yield {"type": "text", "data": data}
                else:
                    prompt_tokens, out_tokens, thinking_tokens = (
                        gemini_rest.usage_tokens(data))
        else:
            response = await _gemini_generate(full_prompt, stream=True)

            async for chunk in response:
                # Τα usage metadata (tokens) έρχονται συνήθως στο τελευταίο chunk
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", None)
                    out_tokens = getattr(usage, "candidates_token_count", None)
                # ΟΧΙ chunk.text σκέτο: σκάει σε thinking-only chunks (βλ. _safe_chunk_text)
                text = _safe_chunk_text(chunk)
                if text:
                    produced_text = True
                    yield {"type": "text", "data": text}

        # Edge case 2.5-flash: όλα τα tokens πήγαν σε «thinking», 0 σε απάντηση ->
        # μη γυρνάς κενό. Μήνυμα ασφαλείας (σπάνιο & διακοπτόμενο -> ξαναδοκίμασε).
        if not produced_text:
            logger.warning("Gemini: κενή απάντηση (thinking-only, finish_reason=STOP).")
            yield {"type": "text",
                   "data": ("Δεν κατάφερα να συνθέσω απάντηση αυτή τη στιγμή — "
                            "δοκίμασε ξανά την ερώτηση." if _has_greek(question)
                            else "I couldn't compose an answer just now — "
                                 "please try asking again.")}
    except asyncio.CancelledError:
        logger.warning(
            "Ο χρήστης έκλεισε τη σύνδεση (Client Disconnected). Διακοπή streaming από το Gemini για προστασία πόρων (FinOps)!")
        raise  # Ενημερώνει το FastAPI να κλείσει το socket
    except Exception as e:
        # Π.χ. επίμονο rate limit (429): υποχωρούμε ομαλά αντί να κρασάρει το stream
        logger.error(f"Σφάλμα παραγωγής απάντησης από το Gemini: {e}")
        metrics.inc("rag_gemini_errors_total", {"type": type(e).__name__})
        metrics.inc("rag_answers_total", {"outcome": "generation_error"})
        yield {"type": "text",
               "data": ("⚠️ Προσωρινό πρόβλημα με το AI (πιθανόν όριο ρυθμού). "
                        "Δοκίμασε ξανά σε λίγο." if _has_greek(question)
                        else "⚠️ Temporary AI issue (possibly a rate limit). "
                             "Please try again shortly.")}

        # --- MLOps lite: latency ανά φάση + FinOps token counting (ποτέ fatal) ---
    # Τα ΙΔΙΑ νούμερα πάνε και στα logs και στο UI: observability που δεν βλέπει
    # ο χρήστης είναι observability που κανείς δεν κοιτά.
    try:
        generation_time = time.perf_counter() - t_generation
        metrics.observe("rag_generation_seconds", generation_time)
        metrics.inc("rag_answers_total", {"outcome": "ok"})
        for kind, n in (("prompt", prompt_tokens), ("completion", out_tokens),
                        ("thinking", thinking_tokens)):
            if n:
                metrics.inc("rag_tokens_total", {"kind": kind}, float(n))
        logger.info(
            f"RAG METRICS | retrieval={retrieval_time:.2f}s "
            f"generation={generation_time:.2f}s | "
            f"tokens(prompt={prompt_tokens or '?'}, completion={out_tokens or '?'}, "
            f"thinking={thinking_tokens if thinking_tokens is not None else '?'})")
        yield {"type": "metrics", "data": {
            "retrieval_s": round(retrieval_time, 2),
            "generation_s": round(generation_time, 2),
            "pages": len(top_3_data),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": out_tokens,
            "thinking_tokens": thinking_tokens,
        }}
        
    except Exception as e:
        # Ποτέ fatal: η απάντηση έχει ήδη σταλεί, τα metrics είναι bonus.
        logger.warning(f"--- Αποτυχία υπολογισμού metrics: {e} ---")


def _normalize_pdf_text(text: str) -> str:
    """Καθαρίζει δύο ζημιές που κάνει η εξαγωγή κειμένου από PDF.

    1. LIGATURES: το pypdf κρατά τα ﬁ ﬂ ﬀ ﬃ ως ΕΝΑΝ χαρακτήρα, οπότε το
       "eﬃcient" του κειμένου δεν ταιριάζει ΠΟΤΕ με το "efficient" της
       ερώτησης — ούτε στο BM25 ούτε στο embedding. Μετρήθηκαν 721 στο corpus.
    2. ΣΥΛΛΑΒΙΣΜΟΣ: λέξεις κομμένες στο τέλος γραμμής ("serv-\\nerless").
       Μετρήθηκαν 941.

    NFKC και ΟΧΙ NFKD: το NFKD αποσυνθέτει τους ελληνικούς τόνους
    (ά -> α + combining accent) και θα έσπαγε το ελληνικό σκέλος. Το NFKC
    κάνει την ίδια compatibility ανάλυση αλλά ΞΑΝΑΣΥΝΘΕΤΕΙ — ελεγμένο.

    ΔΕΝ λύνει τα κενά ΜΕΣΑ σε λέξεις ("distribu ted", "A WS"): εκεί δεν
    ξέρεις ποιο κενό είναι πραγματικό. Αυτό θέλει καλύτερο extractor.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"([a-zA-Zα-ωΑ-Ω])-\n([a-zα-ω])", r"\1\2", text)


def ingest_pdf(file_path: str, filename: str, user_id: int,
               is_public: bool = False, doc_id: int | None = None) -> bool:
    """Διαβάζει το PDF ανά σελίδα, το κόβει σε chunks και τα αποθηκεύει ΜΑΖΙΚΑ
    (batch) στη ChromaDB μαζί με τον ιδιοκτήτη (user_id) για authorization."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,   # νικητής chunk-experiment: MRR 0.803, nDCG 0.821, coverage 95%
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],  # σεβαστεί παραγράφους/προτάσεις
    )

    documents, metadatas, ids = [], [], []
    # pymupdf: ενώνει σωστά τα text spans του PDF, ενώ το pypdf έβαζε κενά μέσα
    # στις λέξεις ("A WS", "distribu ted") -> λέξεις που δεν υπήρχαν πουθενά.
    with pymupdf.open(file_path) as doc:
        for page_idx, page in enumerate(doc):
            text = _normalize_pdf_text(page.get_text() or "")
            if not text or not text.strip():
                continue
            for chunk_idx, chunk in enumerate(splitter.split_text(text)):
                documents.append(chunk)
                metadatas.append({
                    "file_name": filename,
                    "page": page_idx + 1,
                    "user_id": user_id,
                    "is_public": is_public,
                    "doc_id": doc_id if doc_id is not None else -1,
                })
                ids.append(
                    f"{filename}_p{page_idx}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")

    if documents:
        # Ένα batch insert αντί για ένα-ένα: δραματικά γρηγορότερο.
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        _bump_corpus_version()  # invalidate το BM25 cache (νέα chunks)
        logger.success(
            f"---> Ingest '{filename}': {len(documents)} chunks (user={user_id}).")
    else:
        logger.warning(
            f"---> Το '{filename}' δεν παρήγαγε κείμενο (πιθανώς σκαναρισμένο PDF).")
    return True
