"""README charts from the results CSVs: model comparison and cold-start.
    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/figures")
MODELS = ["popularity", "bpr", "item_knn", "als"]
LABELS = {"popularity": "Popularity", "bpr": "BPR", "item_knn": "Item-KNN", "als": "ALS"}
BLUE, GREY = "#2b6cb0", "#a0aec0"


def bars(ax, labels, values, colors, title, fmt="{:.3f}"):
    b = ax.barh(labels, values, color=colors)
    ax.bar_label(b, labels=[fmt.format(v) for v in values], padding=3)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv("results/results.csv").set_index("model").loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, col, title in zip(axes, ["recall@20", "ndcg@20"], ["Recall@20", "NDCG@20"]):
        bars(ax, [LABELS[m] for m in MODELS], res[col].to_numpy(), [GREY, GREY, GREY, BLUE], title, "{:.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=130)

    cold = pd.read_csv("results/cold_start.csv").set_index("model")
    order = ["random", "content", "tag_to_als"]
    names = {"random": "Random", "content": "Tag similarity", "tag_to_als": "Tags -> ALS space"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
    for ax, col, title in zip(axes, ["recall@20", "ndcg@20"], ["Recall@20", "NDCG@20"]):
        bars(ax, [names[m] for m in order], cold.loc[order, col].to_numpy(), [GREY, GREY, BLUE], title, "{:.4f}")
    fig.suptitle("Cold start: ranking tracks with zero training plays", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "cold_start.png", dpi=130)


if __name__ == "__main__":
    main()
