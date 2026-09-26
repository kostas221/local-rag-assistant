"""RAGAS evaluation με κριτή ΔΕΥΤΕΡΗΣ οικογένειας μοντέλων (OpenAI), πάνω στο
ΙΔΙΟ ragas_dataset.jsonl με το run_ragas.py.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ: το run_ragas.py κρίνει με Gemini, δηλαδή με το ίδιο μοντέλο που
παρήγαγε τις απαντήσεις. Δύο από τις τέσσερις μετρικές (faithfulness,
answer_relevancy) συγκρίνουν κείμενο ΤΟΥ Gemini, οπότε είναι εκτεθειμένες σε
αυτοπροτίμηση. Οι άλλες δύο (context_precision, context_recall) συγκρίνουν την
ανθρώπινη ερώτηση και το ανθρώπινο reference_answer με ακατέργαστο κείμενο PDF
και δεν είναι — μένει όμως το ενδεχόμενο γενικής επιείκειας του κριτή. Και τα
δύο τα ελέγχει η αλλαγή οικογένειας.

ΔΕΝ αγγίζει το pipeline: το dataset (ερωτήσεις, contexts, απαντήσεις) είναι ήδη
παραγμένο. Αλλάζει ΜΟΝΟ ο κριτής.

Setup: το langchain-openai είναι ΗΔΗ στο ragas_env (1.4.1, openai 2.50.0) — δεν
χρειάζεται εγκατάσταση. Λείπει μόνο το κλειδί:
    # στο .env του repo: OPENAI_API_KEY=sk-...
Δοκιμασμένο με ragas 0.4.3 (οι παλιές εισαγωγές μετρικών δουλεύουν, με
DeprecationWarning).

Εκτέλεση:
    # 1) ΠΡΩΤΑ smoke test σε 3 ερωτήσεις — κόστος ~3% του συνόλου.
    ragas_env\\Scripts\\python backend\\evaluation\\run_ragas_alt.py --limit 3
    # 2) Αν το CSV έχει 4 αριθμούς σε κάθε γραμμή (όχι NaN), τρέξε ολόκληρο:
    ragas_env\\Scripts\\python backend\\evaluation\\run_ragas_alt.py

Όγκος ολόκληρου τρεξίματος: 1.814.432 χαρακτήρες contexts ≈ 454k input tokens
ανά πέρασμα. Οι faithfulness / context_precision / context_recall περνούν τα
contexts από μία φορά η καθεμία, το answer_relevancy όχι — σύνολο ~1,4M input
tokens. Υπολόγισε το κόστος με την τρέχουσα τιμή πριν το τρέξεις.
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent      # backend/evaluation
ROOT = HERE.parent.parent                   # root του repo
DATASET = HERE / "ragas_dataset.jsonl"
OUT_CSV = HERE / "runs" / "ragas_results_openai.csv"

# Ο κριτής. temperature=0: ίδια σύμβαση με το run_ragas.py, ώστε η μόνη
# μεταβλητή που αλλάζει μεταξύ των δύο τρεξιμάτων να είναι η οικογένεια.
JUDGE_MODEL = os.getenv("RAGAS_ALT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("RAGAS_ALT_EMBED", "text-embedding-3-small")

load_dotenv(ROOT / ".env")
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("Λείπει OPENAI_API_KEY από το .env του repo.")

# --- SHIM για το ragas >=0.2 πάνω σε langchain-community >=0.4 ---------------
# Ίδιο με το run_ragas.py και ΑΝΕΞΑΡΤΗΤΟ από τον πάροχο: το ragas κάνει
# top-level import του `langchain_community.chat_models.vertexai.ChatVertexAI`,
# που αφαιρέθηκε στο langchain-community 0.4 και σκάει στο module load.
import langchain_community.chat_models as _cm  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Dataset  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

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


def _evaluate(rows, llm, emb):
    """Ένα evaluate() πάνω σε λίστα από rows του dataset -> DataFrame."""
    ds = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "contexts":     [r["contexts"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        # max_workers=4: το rate limit του OpenAI είναι πιο ανεκτικό από του
        # Gemini free tier. timeout ψηλά ώστε ένα αργό call να μην ξαναχρεωθεί
        # σε retry.
        run_config=RunConfig(max_workers=4, timeout=300),
    )
    return result.to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="τρέξε μόνο τις Ν πρώτες ερωτήσεις (smoke test)")
    ap.add_argument("--batch", type=int, default=9,
                    help="ερωτήσεις ανά παρτίδα· κάθε παρτίδα σώζεται ΧΩΡΙΣΤΑ")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Ίδιο φιλτράρισμα με το run_ragas.py: οι out-of-corpus έχουν κενά contexts
    # (τις έκοψε το gate) και ελέγχουν το gate, όχι την ποιότητα RAG.
    in_corpus = [r for r in rows if r["contexts"]]
    if args.limit:
        in_corpus = in_corpus[:args.limit]

    chars = sum(len("".join(r["contexts"])) for r in in_corpus)
    print(f"κριτής: {JUDGE_MODEL} | embeddings: {EMBED_MODEL}")
    print(f"{len(in_corpus)} ερωτήσεις | {chars:,} χαρ contexts "
          f"(~{chars // 4:,} tokens ανά πέρασμα)")
    if args.limit:
        print("*** SMOKE TEST — έλεγξε ότι δεν βγαίνουν NaN πριν τρέξεις ολόκληρο ***")

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))

    out = OUT_CSV if not args.limit else OUT_CSV.with_name("ragas_smoke_openai.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # ΓΙΑΤΙ ΠΑΡΤΙΔΕΣ: το evaluate() δεν κρατάει τίποτα ενδιάμεσα. Μια αποτυχία
    # στην 40ή ερώτηση θα έσβηνε 39 πληρωμένες κλήσεις. Κάθε παρτίδα σώζεται
    # χωριστά στο parts/ και ό,τι υπάρχει ήδη ΔΕΝ ξαναζητιέται — άρα ένα δεύτερο
    # τρέξιμο μετά από crash συνεχίζει, δεν ξαναχρεώνει.
    parts_dir = out.parent / "parts"
    parts_dir.mkdir(exist_ok=True)
    batches = [in_corpus[i:i + args.batch]
               for i in range(0, len(in_corpus), args.batch)]

    frames = []
    for i, batch in enumerate(batches, 1):
        part = parts_dir / f"{out.stem}_part{i}.csv"
        if part.exists():
            print(f"[{i}/{len(batches)}] υπάρχει ήδη, παραλείπεται -> {part.name}")
            frames.append(pd.read_csv(part))
            continue
        print(f"[{i}/{len(batches)}] {len(batch)} ερωτήσεις...")
        df = _evaluate(batch, llm, emb)
        df.to_csv(part, index=False, encoding="utf-8-sig")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")   # BOM: ανοίγει σωστά σε Excel

    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 56)
    print(f"RAGAS — κριτής {JUDGE_MODEL} | {len(df)} ερωτήσεις, μέσοι όροι")
    print("=" * 56)
    for c in cols:
        print(f"  {c:<18} {df[c].mean():.4f}")

    nan = int(df[cols].isna().sum().sum())
    print(f"\nΑνά ερώτηση: {out}")
    print(f"κενά (NaN) κελιά: {nan}"
          + ("  <-- ΠΡΟΒΛΗΜΑ parsing" if nan else "  — καθαρό"))


if __name__ == "__main__":
    main()
