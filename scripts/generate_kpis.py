import matplotlib.pyplot as plt
import numpy as np

# Ρυθμίσεις εμφάνισης
plt.style.use("seaborn-v0_8-whitegrid")
teal_color = "#117a65"
dark_color = "#2c3e50"


# --- 1. KPI Dashboard (Κάρτες Σύνοψης) ---
def create_kpi_dashboard():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    kpis = [
        ("MRR (IN-CORPUS)", "0.793", "×5.3 πάνω από τυχαίο"),
        ("KEYWORD COVERAGE", "98.5%", "κλείνει 98% του περιθωρίου"),
        ("ΠΟΙΟΤΗΤΑ", "4.94/5", "48/50 τέλειες απαντήσεις"),
        ("ΑΜΥΝΑ ΨΕΥΔΑΙΣΘΗΣΗΣ", "5/5", "0 ψευδαισθήσεις"),
        ("LATENCY", "2.8s", "retrieval ~0.85s"),
    ]

    for i, (title, value, sub) in enumerate(kpis):
        x_center = 0.1 + i * 0.2

        # Δημιουργία περιγράμματος (κάρτας)
        rect = plt.Rectangle(
            (x_center - 0.09, 0.1),
            0.18,
            0.8,
            fill=True,
            color="white",
            ec="#d0d3d4",
            lw=1.5,
            zorder=1,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Προσθήκη κειμένων
        ax.text(
            x_center,
            0.75,
            title,
            ha="center",
            va="center",
            fontsize=9,
            color="#7f8c8d",
            weight="bold",
            transform=ax.transAxes,
            zorder=2,
        )
        ax.text(
            x_center,
            0.5,
            value,
            ha="center",
            va="center",
            fontsize=22,
            color=teal_color,
            weight="bold",
            transform=ax.transAxes,
            zorder=2,
        )
        ax.text(
            x_center,
            0.25,
            sub,
            ha="center",
            va="center",
            fontsize=8,
            color="#95a5a6",
            transform=ax.transAxes,
            zorder=2,
        )

    plt.tight_layout()
    plt.savefig("5_kpi_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


# --- 2. LLM Judge Quality (Γράφημα Σύγκρισης) ---
def plot_llm_judge():
    metrics = ["Accuracy", "Completeness", "Relevance", "Faithfulness"]
    without_mh = [5.00, 4.98, 5.00, 5.00]
    all_queries = [4.94, 4.90, 4.96, 4.94]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(
        x - width / 2, without_mh, width, label="Χωρίς Multi-hop (46)", color=teal_color
    )
    rects2 = ax.bar(
        x + width / 2, all_queries, width, label="Όλες (50)", color=dark_color
    )

    ax.set_ylabel("Βαθμολογία LLM-Judge (0-5)", fontsize=12)
    ax.set_title(
        "Ποιότητα Απάντησης — Επίδραση των Multi-hop Ερωτήσεων", fontsize=14, pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")

    # Περιορισμός του άξονα Y για να γίνουν εμφανείς οι μικρές διαφορές
    ax.set_ylim(4.8, 5.05)
    ax.legend(loc="lower left")

    # Προσθήκη των ακριβών αριθμών πάνω από κάθε ράβδο
    for r in [rects1, rects2]:
        for bar in r:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.005,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("6_llm_judge_quality.png", dpi=300)
    plt.close()


# Εκτέλεση
if __name__ == "__main__":
    create_kpi_dashboard()
    plot_llm_judge()
    print(
        "Δημιουργήθηκαν τα γραφήματα '5_kpi_dashboard.png' και '6_llm_judge_quality.png'!"
    )
