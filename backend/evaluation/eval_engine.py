import asyncio
import math
import json
import os
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# --- ΕΙΣΑΓΩΓΗ ΑΠΟ ΤΟ ΔΙΚΟ ΣΟΥ BACKEND ---
from ai_core import search_documents, ask_ai

load_dotenv()

# Ορίζουμε το μοντέλο που θα κάνει το "judging" με την επίσημη βιβλιοθήκη της Google
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# Το golden_set_20 αφορά ΑΠΟΚΛΕΙΣΤΙΚΑ αυτά τα 2 papers (10 Q «Above the Clouds»,
# 8 Q «Serverless», 2 αρνητικά out-of-corpus). Καρφώνουμε το eval corpus εδώ ώστε
# τυχόν ΑΛΛΑ έγγραφα στη βάση (π.χ. demo uploads) να ΜΗΝ μολύνουν τη μέτρηση — το
# αποτέλεσμα μένει reproducible πάνω στο documented corpus του RESULTS.md.
GOLDEN_CORPUS = ["EECS-2009-28.pdf", "1902.03383v1.pdf"]


class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str


class RetrievalEval(BaseModel):
    mrr: float
    ndcg: float
    keyword_coverage: float


class AnswerEval(BaseModel):
    feedback: str = Field(description="Αιτιολόγηση της βαθμολογίας")
    accuracy: float = Field(description="Ακρίβεια (1-5)")
    completeness: float = Field(description="Πληρότητα (1-5)")
    relevance: float = Field(description="Σχετικότητα (1-5)")
    faithfulness: float = Field(
        description="Πιστότητα στις πηγές / απουσία ψευδαισθήσεων (1-5)")

# --- Υπολογισμός Metrics ---


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in doc.lower(
    ) else 0 for doc in retrieved_docs[:k]]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

# --- ΟΙ ΔΥΟ ΚΥΡΙΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ΑΞΙΟΛΟΓΗΣΗΣ ---


async def evaluate_retrieval(test: TestQuestion) -> RetrievalEval:
    results = await search_documents(test.question, target_filenames=GOLDEN_CORPUS)
    retrieved_texts = [text for text, meta in results]

    if not retrieved_texts:
        return RetrievalEval(mrr=0.0, ndcg=0.0, keyword_coverage=0.0)

    mrr_scores = [calculate_mrr(kw, retrieved_texts) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(kw, retrieved_texts, k=3)
                   for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    coverage = (keywords_found / total_keywords *
                100) if total_keywords > 0 else 0.0

    return RetrievalEval(mrr=avg_mrr, ndcg=avg_ndcg, keyword_coverage=coverage)


async def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str]:
    # 1. Φέρνουμε το ίδιο context που θα έβλεπε το μοντέλο (για το faithfulness)
    retrieved = await search_documents(test.question, target_filenames=GOLDEN_CORPUS)
    context = "\n---\n".join(text for text, meta in retrieved)

    # 2. Παράγουμε την απάντηση του συστήματος (ίδιο pinned corpus)
    full_answer = ""
    async for chunk in ask_ai(test.question, target_filenames=GOLDEN_CORPUS):
        if chunk["type"] == "text":
            full_answer += chunk["data"]

    # 3. Ο "Δικαστής" αξιολογεί — ΣΥΜΠΕΡΙΛΑΜΒΑΝΟΝΤΑΣ το faithfulness ως προς το context
    prompt = f"""You are a strict expert evaluator for a RAG system. Only give 5/5 for perfect answers.

QUESTION: {test.question}

RETRIEVED CONTEXT (the ONLY source the system was allowed to use):
{context}

GENERATED ANSWER: {full_answer}

REFERENCE ANSWER (gold): {test.reference_answer}

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

    try:
        response = await judge_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0)  # ντετερμινιστικός judge -> επαναλήψιμα αποτελέσματα
        )
        data = json.loads(response.text)
        answer_eval = AnswerEval(**data)
    except Exception as e:
        print(f"Σφάλμα κατά το judging: {e}")
        answer_eval = AnswerEval(
            feedback="Σφάλμα API", accuracy=0, completeness=0,
            relevance=0, faithfulness=0)

    return answer_eval, full_answer
