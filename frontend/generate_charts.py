import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Ορισμός κοινού στυλ για επαγγελματική εμφάνιση
plt.style.use("seaborn-v0_8-whitegrid")
colors = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71"]


# --- 1. Η Θετική Σκάλα (Hybrid RAG MRR) ---
def plot_hybrid_mrr():
    stages = ["Dense Only", "+ BM25 / RRF", "+ Cross-Encoder", "+ Relevance Gate"]
    mrr_values = [0.731, 0.774, 0.793, 0.793]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(stages, mrr_values, color=colors[0])
    plt.title("Επίδραση Υβριδικής Μηχανικής στο MRR", fontsize=14, pad=15)
    plt.ylabel("MRR Score", fontsize=12)
    plt.ylim(0.65, 0.85)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            f"{yval:.3f}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("1_hybrid_mrr.png", dpi=300)
    plt.close()


# --- 2. Ανάκτηση ανά Κατηγορία ---
def plot_category_mrr():
    categories = [
        "Γεγονότος",
        "Αιτιολόγησης",
        "Απαρίθμησης",
        "Πολλαπλών βημάτων",
    ]
    mrr = [0.869, 0.831, 0.751, 0.541]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(categories, mrr, color=colors[2])
    plt.ylabel("MRR", fontsize=12)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    )

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.3f}".replace(".", ","),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("2_category_mrr.png", dpi=300)
    plt.close()


# --- 3. RAGAS Metrics (Ανεξάρτητος Κριτής) ---
def plot_ragas_metrics():
    metrics = [
        "Πιστότητα",
        "Ανάκληση συμφραζομένων",
        "Συνάφεια απάντησης",
        "Ακρίβεια συμφραζομένων",
    ]
    scores = [0.982, 1.000, 0.857, 0.778]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(metrics, scores, color=colors[3])
    plt.xlabel("Βαθμολογία", fontsize=12)
    plt.xlim(0.6, 1.05)
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    )

    for bar in bars:
        xval = bar.get_width()
        plt.text(
            xval + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{xval:.3f}".replace(".", ","),
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("3_ragas_metrics.png", dpi=300)
    plt.close()


# --- 4. Relevance Threshold (Min-P Margin) ---
def plot_relevance_gap():
    labels = ["Max Out-of-Corpus", "Threshold", "Min In-Corpus"]
    values = [-3.12, -2.60, -2.08]

    plt.figure(figsize=(7, 4))
    plt.hlines(y=0, xmin=-4, xmax=-1, color="gray", linestyle="--", alpha=0.5)
    plt.plot(values, [0, 0, 0], "o-", markersize=10, color=colors[1], linewidth=2)

    for i, (label, val) in enumerate(zip(labels, values)):
        plt.text(
            val,
            0.02,
            f"{label}\n({val})",
            ha="center",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.title("Περιθώριο Ασφαλείας (Logits Gap)", fontsize=14, pad=20)
    plt.yticks([])
    plt.xlim(-4, -1)

    plt.tight_layout()
    plt.savefig("4_relevance_gap.png", dpi=300)
    plt.close()


# Εκτέλεση συναρτήσεων
if __name__ == "__main__":
    plot_hybrid_mrr()
    plot_category_mrr()
    plot_ragas_metrics()
    plot_relevance_gap()
    print("Τα διαγράμματα δημιουργήθηκαν επιτυχώς στο τρέχον directory!")
