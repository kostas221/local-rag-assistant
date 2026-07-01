"""Streamlit dashboard για τα eval αποτελέσματα — ΔΙΑΒΑΖΕΙ το υπάρχον results.csv
(ΧΩΡΙΣ re-run, μηδέν κόστος). Δείχνει χρωματισμένες μετρικές + breakdown ανά
κατηγορία & γλώσσα. Τρέξε:  streamlit run evaluation/dashboard.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
RESULTS = HERE / "results.csv"
GOLDEN = HERE / "golden_set_20.jsonl"

st.set_page_config(page_title="Z-AI RAG Evaluation", layout="wide")
st.title("📊 Z-AI Platform — RAG Evaluation Dashboard")
st.caption("Διαβάζει το `results.csv` του τελευταίου run — δεν ξανατρέχει αξιολόγηση.")

if not RESULTS.exists():
    st.error(f"Δεν βρέθηκε {RESULTS}. Τρέξε πρώτα: `run_eval.py golden_set_20.jsonl`")
    st.stop()

df = pd.read_csv(RESULTS, encoding="utf-8-sig")

# Categories από το golden set (match με βάση το κείμενο της ερώτησης)
cat_map = {}
if GOLDEN.exists():
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cat_map[d["question"]] = d.get("category", "uncategorized")
df["category"] = df["question"].map(cat_map).fillna("uncategorized")
df["lang"] = df["question"].apply(
    lambda q: "EL" if any('Ͱ' <= c <= 'Ͽ' or 'ἀ' <= c <= '῿'
                          for c in str(q)) else "EN")

# In-corpus = όχι out_of_corpus (το MRR ορίζεται μόνο εκεί)
inc = df[df["category"] != "out_of_corpus"]


def _color(val, green, amber):
    return "#28a745" if val >= green else ("#fd7e14" if val >= amber else "#dc3545")


def card(label, val, green, amber, fmt="{:.3f}"):
    c = _color(val, green, amber)
    st.markdown(
        f"""<div style="padding:14px;border-radius:8px;border-left:6px solid {c};
        background:#1e1e1e;margin-bottom:8px;">
        <div style="color:#aaa;font-size:13px">{label}</div>
        <div style="color:{c};font-size:30px;font-weight:bold">{fmt.format(val)}</div>
        </div>""", unsafe_allow_html=True)


st.subheader("🔎 Retrieval — in-corpus (18 ερωτήσεις)")
c1, c2, c3 = st.columns(3)
with c1:
    card("MRR", inc["mrr"].mean(), 0.85, 0.70)
with c2:
    card("nDCG", inc["ndcg"].mean(), 0.85, 0.70)
with c3:
    card("Keyword Coverage", inc["keyword_coverage"].mean(), 90, 75, "{:.1f}%")

st.subheader("💬 Answer quality — LLM-as-judge (/5)")
cols = st.columns(4)
for col, m in zip(cols, ["accuracy", "completeness", "relevance", "faithfulness"]):
    with col:
        card(m.capitalize(), df[m].mean(), 4.5, 4.0, "{:.2f}/5")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("MRR ανά κατηγορία")
    st.bar_chart(df.groupby("category")["mrr"].mean())
with col_b:
    st.subheader("MRR ανά γλώσσα (in-corpus)")
    st.bar_chart(inc.groupby("lang")["mrr"].mean())

# Anti-hallucination: out-of-corpus πρέπει να δίνουν 0 retrieval (gate)
ooc = df[df["category"] == "out_of_corpus"]
if not ooc.empty:
    st.subheader("🛡️ Anti-hallucination (out-of-corpus — πρέπει να απορρίπτονται)")
    st.dataframe(ooc[["question", "mrr", "keyword_coverage", "faithfulness"]],
                 use_container_width=True)

st.subheader("📋 Αναλυτικά ανά ερώτηση")
st.dataframe(
    df[["category", "lang", "mrr", "keyword_coverage", "accuracy",
        "completeness", "relevance", "faithfulness", "question"]],
    use_container_width=True)
