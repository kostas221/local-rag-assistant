"""Επίσημο RAGAS evaluation πάνω στο ragas_dataset.jsonl (το παράγει το
faithfulness_eval.py με το τρέχον pipeline). Τρέχει ΕΚΤΟΣ backend container:
το RAGAS θέλει pydantic v2, ασύμβατο με το pinned stack (Pydantic V1) του backend.

Setup (μία φορά, από το root του repo):
    python -m venv ragas_env
    ragas_env\\Scripts\\pip install "ragas==0.2.15" langchain-google-genai datasets python-dotenv

Εκτέλεση:
    ragas_env\\Scripts\\python backend\\evaluation\\run_ragas.py

Μετρικές (judge: gemini-2.5-flash @ temperature=0 — ίδια σύμβαση με το custom harness):
- faithfulness      : στηρίζεται η απάντηση ΜΟΝΟ στα contexts; (anti-hallucination)
- answer_relevancy  : απαντά όντως στην ερώτηση; (embeddings: text-embedding-004)
- context_precision : τα σχετικά contexts κατατάσσονται ψηλά;
- context_recall    : τα contexts καλύπτουν το ground-truth answer;

Οι out-of-corpus ερωτήσεις (κενά contexts — τις έκοψε το relevance gate) εξαιρούνται:
ελέγχουν το gate, όχι την ποιότητα RAG (ίδια σύμβαση με τα in-corpus του RESULTS.md).
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent      # backend/evaluation
ROOT = HERE.parent.parent                   # root του repo
DATASET = HERE / "ragas_dataset.jsonl"
OUT_CSV = HERE / "ragas_results.csv"

# Το .env του repo έχει GEMINI_API_KEY· το langchain-google-genai περιμένει GOOGLE_API_KEY.
load_dotenv(ROOT / ".env")
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

# Imports μετά το env setup (κάποια libs διαβάζουν το key στο import)
# --- SHIM για το ragas >=0.2 πάνω σε langchain-community >=0.4 ---------------
# Το ragas κάνει top-level import του `langchain_community.chat_models.vertexai.
# ChatVertexAI`, που ΑΦΑΙΡΕΘΗΚΕ στο langchain-community 0.4. Δεν το χρησιμοποιούμε
# πουθενά (ο judge είναι Gemini μέσω langchain-google-genai), αλλά το import σκάει
# στο module load και μπλοκάρει ολόκληρο το ragas.
#
# ΓΙΑΤΙ SHIM ΚΑΙ ΟΧΙ DOWNGRADE: το ragas 0.2.15 δηλώνει `langchain-community`
# ΧΩΡΙΣ version pin, οπότε το pip φέρνει πάντα το τελευταίο. Καρφώνοντας παλιό
# langchain-community σέρνει downgrade και σε core/langchain/google-genai — τέσσερα
# πακέτα σε εκδόσεις του 2024, με το gemini-embedding-001 να μην υποστηρίζεται.
# Το stub είναι 5 γραμμές και σπάει ΜΟΝΟ ό,τι δεν χρησιμοποιούμε.
import types  # noqa: E402

import langchain_community.chat_models as _cm  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_google_genai import (  # noqa: E402
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

if not hasattr(_cm, "vertexai"):
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})   # ποτέ δεν instantiate-άρεται
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
    _cm.vertexai = _stub

from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402


def main():
    rows = [json.loads(line) for line in
            DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    in_corpus = [r for r in rows if r["contexts"]]
    print(f"{len(rows)} ερωτήσεις | {len(in_corpus)} in-corpus | "
          f"{len(rows) - len(in_corpus)} out-of-corpus (gate) εξαιρούνται")

    ds = Dataset.from_dict({
        "question":     [r["question"] for r in in_corpus],
        "contexts":     [r["contexts"] for r in in_corpus],
        "answer":       [r["answer"] for r in in_corpus],
        "ground_truth": [r["ground_truth"] for r in in_corpus],
    })

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        # gemini-embedding-001: το τρέχον GA embedding μοντέλο — το παλιό
        # text-embedding-004 αποσύρθηκε (404) αρχές 2026.
        model="models/gemini-embedding-001"))

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        # max_workers=2: ήπιο στο per-minute rate limit του Gemini
        run_config=RunConfig(max_workers=2, timeout=180),
    )

    print("\n" + "=" * 56)
    print("RAGAS ΑΠΟΤΕΛΕΣΜΑΤΑ (in-corpus, μέσοι όροι)")
    print("=" * 56)
    print(result)

    df = result.to_pandas()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # BOM: ανοίγει σωστά σε Excel
    print(f"\n✅ Ανά ερώτηση: {OUT_CSV}")


if __name__ == "__main__":
    main()
