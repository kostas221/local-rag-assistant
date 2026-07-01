import re
import time
import unicodedata
import os
import uuid
import asyncio
import threading
import torch
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from google.api_core import exceptions as gexc

from dotenv import load_dotenv
import google.generativeai as genai
from loguru import logger

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

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device=DEVICE
)

collection = chroma_client.get_or_create_collection(
    name="ai_research_docs",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"},  # bge-m3 είναι φτιαγμένο για cosine (default της Chroma είναι L2)
)

logger.info("---> Φόρτωση του Reranker (Αξιολογητή)...")
# Multilingual reranker BAAI/bge-reranker-v2-m3 (βάση XLM-R-large): πολύ ισχυρό
# σε πολλές γλώσσες (μαζί ελληνικά). Δίνει score sigmoid 0-1 (βλ. relevance gate παρακάτω).
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=DEVICE)

# --- Relevance gate threshold (anti-hallucination, 1η από 2 γραμμές άμυνας) ---
# bge-reranker-v2-m3: score sigmoid 0-1. Βαθμονομήθηκε στο golden_set_20:
# out-of-corpus (quantum/bitcoin) ~0.00, ενώ ΜΕΤΑΦΡΑΣΜΕΝΕΣ ελληνικές ερωτήσεις
# σκοράρουν χαμηλά (elasticity 0.25, cold-start 0.30) -> το 0.30 τις έκοβε.
# 0.15: τις περνά, μπλοκάρει τα άσχετα. 2η γραμμή άμυνας = το system prompt
# ("αν δεν είναι στο κείμενο, πες το") -> faithfulness 4.95/5 το επιβεβαιώνει.
# Module-level: βαθμονομείται/δοκιμάζεται σε ΕΝΑ σημείο (βλ. tests/test_rag_core.py).
MIN_RERANK_SCORE = 0.15
# Πόσα candidates (από το RRF) περνάνε στον reranker — ο ΑΚΡΙΒΟΣ βήμα στη CPU.
# Μετρημένο (golden in-corpus): 10->15 ανέβασε coverage 93.5%->97.2%, MRR
# 0.824->0.846 (διορθώθηκε η ExCamera/Q11)· >15 = plateau. Trade-off: ~+60%
# χρόνος rerank στη CPU. Env-configurable: σε CPU production υπό φόρτο -> "10".
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))


def delete_file_from_db(filename: str, user_id: int = None, doc_id: int = None):
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
        logger.error(f"---> Σφάλμα κατά τη διαγραφή από τη Vector DB: {e}")


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
_translation_cache = {}


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
        prompt = (
            "You are a translation assistant for an English-only academic search engine. "
            "Translate the user's question to English and output ONLY a concise English "
            "search query with the key terms. No quotes, no extra text.\n\n"
            f"User question: {query}"
        )
        response = await _gemini_generate(prompt)
        english_query = response.text.strip(' "\'\n')
        _translation_cache[query] = english_query
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
        response = await _gemini_generate(prompt)
        rewritten = _safe_chunk_text(response).strip(' "\'\n')
        if rewritten:
            logger.info(f"---> Query rewrite: '{question[:40]}...' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"--- Query rewrite failed: {e} ---")
    return question


# --- BM25 index cache --------------------------------------------------------
# Το χτίσιμο του BM25 (tokenization ΟΛΟΥ του corpus + BM25Okapi) είναι pure-Python
# και κρατάει το GIL· χωρίς cache γινόταν σε ΚΑΘΕ query -> με πολλούς ταυτόχρονους
# χρήστες οι ερωτήσεις σειριοποιούνταν πάνω σε αυτό. Τώρα χτίζεται ΜΙΑ φορά ανά
# "έκδοση" του corpus και invalidate-άρεται σε ingest/delete.
# ΣΗΜ ΑΣΦΑΛΕΙΑΣ: το index είναι πάνω σε ΟΛΟ το corpus· το authorization (ποιος
# βλέπει τι) ΔΕΝ γίνεται εδώ — μένει single-source στη Chroma (το where του
# get/query) κι εφαρμόζεται μέσω του allowed_ids στο search_documents.
_bm25_lock = threading.Lock()
_bm25_cache = {"version": None}
_corpus_version = 0


def _bump_corpus_version():
    """Invalidate του BM25 cache μετά από ingest/delete (άλλαξε το corpus)."""
    global _corpus_version
    _corpus_version += 1


def _get_bm25_index():
    """Cached (bm25, ids, texts, metas, pos) πάνω σε ΟΛΟ το corpus. Ξαναχτίζεται
    μόνο όταν αλλάξει το _corpus_version. CPU-bound στο (επανα)χτίσιμο -> κάλεσέ
    το μέσα σε asyncio.to_thread."""
    if _bm25_cache.get("version") == _corpus_version:
        return _bm25_cache
    with _bm25_lock:
        # double-check: άλλο thread μπορεί να το έχτισε όσο περιμέναμε το lock
        if _bm25_cache.get("version") == _corpus_version:
            return _bm25_cache
        data = collection.get()  # όλο το corpus (ids + documents + metadatas)
        ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
        tokenized = [el_tokenize(t) for t in texts]
        _bm25_cache.clear()
        _bm25_cache.update(
            version=_corpus_version, ids=ids, texts=texts, metas=metas,
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


def _rrf_fuse(dense_ids, sparse_ids, all_ids, all_texts, all_metadatas,
              k: int = 60, top_n: int = 15):
    """Reciprocal Rank Fusion: ενώνει τα dense + sparse rankings σε ΕΝΑ score
    χρησιμοποιώντας ΜΟΝΟ τη θέση (rank) στη λίστα — άρα δεν χρειάζεται κοινή
    κλίμακα ανάμεσα σε cosine similarity και BM25. Καθαρή (pure) συνάρτηση ->
    δοκιμάζεται χωρίς ChromaDB/μοντέλα.

    k=60: ακαδημαϊκή σταθερά RRF. Doc που λείπει από μία λίστα παίρνει rank 1000
    (σχεδόν μηδενική συνεισφορά) αντί να αποκλείεται εντελώς.
    Επιστρέφει [(rrf_score, text, metadata)] φθίνουσα, κομμένο στα top_n."""
    unique_ids = list(set(dense_ids + sparse_ids))
    dense_ranks = {id_: rank for rank, id_ in enumerate(dense_ids)}
    sparse_ranks = {id_: rank for rank, id_ in enumerate(sparse_ids)}

    fused = []
    for doc_id in unique_ids:
        rank_dense = dense_ranks.get(doc_id, 1000) + 1
        rank_sparse = sparse_ranks.get(doc_id, 1000) + 1
        rrf_score = (1.0 / (k + rank_dense)) + (1.0 / (k + rank_sparse))
        idx = all_ids.index(doc_id)
        fused.append((rrf_score, all_texts[idx], all_metadatas[idx]))

    return sorted(fused, key=lambda x: x[0], reverse=True)[:top_n]


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
    for _score, _text, meta in top_chunks:
        key = (meta.get("doc_id"), meta.get("page"), meta.get("file_name"))
        if key not in seen:
            seen.append(key)
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


def _build_where(target_filenames: list = None, user_id: int = None):
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


async def search_documents(raw_query: str, target_filenames: list = None, user_id: int = None):
    # Αν ο χρήστης δεν έχει επιλέξει κανένα (έγκυρο) αρχείο, μη γυρνάς τυχαία
    # αποτελέσματα — απάντησε "τίποτα".
    if target_filenames is not None and len(target_filenames) == 0:
        return []

    # 0. Κρυφή μετάφραση/εμπλουτισμός (Gemini API)
    query = await optimize_query(raw_query)

    where_filter = _build_where(target_filenames, user_id)

    # 1. Allowed ids (AUTHORIZATION single-source στη Chroma). Παίρνουμε ΜΟΝΟ
    # ids/metas — ΟΧΙ το βαρύ documents· τα κείμενα τα έχει το cached BM25 index.
    allowed = await asyncio.to_thread(
        lambda: collection.get(where=where_filter, include=["metadatas"]))
    allowed_ids = allowed["ids"]
    if not allowed_ids:
        return []

    # Cached BM25 index πάνω σε ΟΛΟ το corpus (χτίζεται μία φορά ανά version).
    idx = await asyncio.to_thread(_get_bm25_index)
    all_ids, all_texts, all_metadatas = idx["ids"], idx["texts"], idx["metas"]

    # 2. Dense Search — bge-m3 (cosine), authz μέσω where. to_thread: CPU-bound embed.
    dense_results = await asyncio.to_thread(
        lambda: collection.query(
            query_texts=[query],
            n_results=min(30, len(allowed_ids)),
            where=where_filter,
        )
    )
    dense_ids = dense_results['ids'][0]

    # 3. Sparse Search (BM25 από το cache), περιορισμένο στα allowed_ids
    sparse_ids = await asyncio.to_thread(
        _bm25_sparse_ids, idx, query, allowed_ids, 30)

    # 4. Reciprocal Rank Fusion (RRF): ένωση dense + sparse -> top-15
    rrf_sorted = _rrf_fuse(dense_ids, sparse_ids, all_ids, all_texts,
                           all_metadatas, k=60, top_n=RERANK_CANDIDATES)

    # 5. Reranking με τον Cross-Encoder — το βαρύτερο CPU κομμάτι, σε thread
    pairs = [[query, item[1]] for item in rrf_sorted]
    cross_scores = await asyncio.to_thread(reranker.predict, pairs)

    final_combined = list(zip(cross_scores, [item[1] for item in rrf_sorted],
                              [item[2] for item in rrf_sorted]))
    sorted_final = sorted(final_combined, key=lambda x: x[0], reverse=True)

    # --- Relevance gate (anti-hallucination) — κατώφλι/σκεπτικό: MIN_RERANK_SCORE ---
    if sorted_final[0][0] < MIN_RERANK_SCORE:
        logger.info(
            f"Relevance gate: best={sorted_final[0][0]:.2f} < {MIN_RERANK_SCORE} "
            f"-> κανένα αρκετά σχετικό chunk")
        return []

    # Parent-document (page-level) expansion: επιστρέφουμε ΟΛΟΚΛΗΡΕΣ ΣΕΛΙΔΕΣ
    # (όχι μεμονωμένα chunks) από τα top reranked chunks -> καλύτερη πληρότητα
    # σε ερωτήσεις τύπου "λίστα όλων των X" (π.χ. όλα τα εμπόδια του cloud).
    return await asyncio.to_thread(_expand_to_pages, sorted_final[:12])

# --- ΜΕΤΑΤΡΟΠΗ ΣΕ ASYNC GENERATOR ---


# Στυλ απάντησης ανά persona (η γλώσσα & ο κανόνας μη-ψευδαίσθησης μένουν σταθερά)
PERSONA_STYLES = {
    "Researcher": "Maintain an objective, scientific tone. Provide a THOROUGH, well-structured answer that fully addresses the question using ALL relevant details from the sources. State the main finding first, then expand with supporting evidence, context and nuances. Use as many sentences as needed for completeness (typically 4-8); do NOT artificially shorten.",
    "Educator": "Explain clearly and simply, as if teaching a student, with a short clarifying example if helpful (3-5 sentences).",
    "Concise": "Be extremely concise: 1-2 sentences or short bullet points, only the essential finding.",
}


async def ask_ai(question, target_filenames, history=None, user_id=None, persona="Researcher"):
    if history is None:
        history = []

    # Conversational query rewriting: σε follow-up (υπάρχει ιστορικό) ξαναγράφουμε
    # την ερώτηση σε αυτόνομο query ΜΟΝΟ για το retrieval. Η ίδια η ερώτηση (question)
    # μένει αναλλοίωτη για τη γέννηση της απάντησης & τη γλώσσα.
    retrieval_query = await _rewrite_query(question, history) if history else question

    # --- MLOps: μέτρηση χρόνου φάσης ανάκτησης (retrieval) ---
    t_retrieval = time.perf_counter()
    top_3_data = await search_documents(retrieval_query, target_filenames, user_id=user_id)
    retrieval_time = time.perf_counter() - t_retrieval

    if not top_3_data:
        yield {"type": "sources", "data": []}
        yield {"type": "text", "data": "No relevant documents found. Make sure you have selected the right PDFs in the left sidebar."}
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
    usage = None
    produced_text = False  # για fallback αν το μοντέλο γυρίσει ΜΟΝΟ thinking
    try:
        response = await _gemini_generate(full_prompt, stream=True)

        async for chunk in response:
            # Τα usage metadata (tokens) έρχονται συνήθως στο τελευταίο chunk
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
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
                   "data": "Δεν κατάφερα να συνθέσω απάντηση αυτή τη στιγμή — δοκίμασε ξανά την ερώτηση."}
    except asyncio.CancelledError:
        logger.warning(
            "Ο χρήστης έκλεισε τη σύνδεση (Client Disconnected). Διακοπή streaming από το Gemini για προστασία πόρων (FinOps)!")
        raise  # Ενημερώνει το FastAPI να κλείσει το socket
    except Exception as e:
        # Π.χ. επίμονο rate limit (429): υποχωρούμε ομαλά αντί να κρασάρει το stream
        logger.error(f"Σφάλμα παραγωγής απάντησης από το Gemini: {e}")
        yield {"type": "text",
               "data": "⚠️ Temporary AI issue (possibly a rate limit). Please try again shortly."}

    # --- MLOps lite: latency ανά φάση + FinOps token counting (ποτέ fatal) ---
    try:
        generation_time = time.perf_counter() - t_generation
        prompt_tokens = getattr(
            usage, "prompt_token_count", "?") if usage else "?"
        out_tokens = getattr(
            usage, "candidates_token_count", "?") if usage else "?"
        logger.info(
            f"RAG METRICS | retrieval={retrieval_time:.2f}s "
            f"generation={generation_time:.2f}s | "
            f"tokens(prompt={prompt_tokens}, completion={out_tokens})")
    except Exception:
        pass


def ingest_pdf(file_path: str, filename: str, user_id: int, is_public: bool = False, doc_id: int = None) -> bool:
    """Διαβάζει το PDF ανά σελίδα, το κόβει σε chunks και τα αποθηκεύει ΜΑΖΙΚΑ
    (batch) στη ChromaDB μαζί με τον ιδιοκτήτη (user_id) για authorization."""
    reader = PdfReader(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,   # νικητής chunk-experiment: MRR 0.803, nDCG 0.821, coverage 95%
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],  # σεβαστεί παραγράφους/προτάσεις
    )

    documents, metadatas, ids = [], [], []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
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
