"""CNN depth search figure from the results of WICOS_AllModels.ipynb (run of 18-10-2026, Jim's pre-processing).

The notebook run stopped after the depth search, so the numbers are copied from its output.
Learning rate 0.03, dropout 0.5; accuracy = mean over the folds (LOSO: 6 persons, random split: 5 folds).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

depth = pd.DataFrame({
    "n_blocks": [4, 5, 6],
    "filters": ["8-16-32-32", "8-16-32-32-32", "8-16-32-32-32-32"],
    "parameters": [23332, 26436, 34148],
    "LOSO train": [0.983, 0.992, 0.986],
    "LOSO validation": [0.672, 0.727, 0.690],
    "Random train": [0.967, 0.989, 0.980],
    "Random validation": [0.913, 0.958, 0.956],
})
selected = int(depth["LOSO validation"].idxmax())          # best LOSO validation accuracy
depth.to_csv(OUT_DIR / "cnn_depth_search.csv", index=False)

with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
    fig, ax = plt.subplots(figsize=(6.4, 3.3), constrained_layout=True)
    x = np.arange(len(depth))
    for k, scheme in enumerate(["LOSO", "Random"]):
        pos = x + (k - 0.5) * 0.36
        ax.bar(pos, depth[f"{scheme} validation"], width=0.34, color=f"C{k}",
               label=f"Validation ({'leave-one-subject-out' if scheme == 'LOSO' else 'random split'})")
        ax.scatter(pos, depth[f"{scheme} train"], marker="_", s=260, linewidths=1.6, color="black",
                   label="Training" if k == 0 else None, zorder=3)
        for p, v in zip(pos, depth[f"{scheme} validation"]):
            ax.text(p, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    ax.axvspan(selected - 0.45, selected + 0.45, color="gold", alpha=0.25, zorder=0, label="Selected (best LOSO)")
    ax.set_xticks(x, [f"{b} blocks\n{f}\n{p / 1000:.1f}k parameters"
                      for b, f, p in zip(depth.n_blocks, depth.filters, depth.parameters)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(labels, key=lambda l: (not l.startswith("Validation"), l == "Training"))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7.5, loc="upper center",
              ncol=2, bbox_to_anchor=(0.5, 1.22), frameon=False)
    fig.savefig(OUT_DIR / "cnn_depth_search.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "cnn_depth_search.png", dpi=150, bbox_inches="tight")
print("saved cnn_depth_search.pdf / .png / .csv")
