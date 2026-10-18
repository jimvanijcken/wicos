"""Report figures for Task 2, from the CSV files written by notebooks/WICOS_AllModels.ipynb.

Usage:  python results/allmodels/make_report_figures.py
Output: report_tuning.pdf       training/validation accuracy vs the key hyperparameter (4 classical models)
        report_comparison.pdf   LOSO, random-split validation and test accuracy of all five models
        report_persons.pdf      LOSO accuracy per person and the confusion matrix of the best LOSO model
The CNN numbers come from cnn_depth_search.csv (the CNN run stopped after the depth search).
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PREVIEW = os.environ.get("PREVIEW_DIR")   # optional folder for PNG previews
STYLE = {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
COLORS = {"LOSO": "C0", "Random": "C1"}


def label(v):
    if isinstance(v, str):
        return v
    return "None" if pd.isna(v) else f"{v:g}"


def best_row(t, scheme):
    r = t[t.scheme == scheme]
    return r.loc[r.val_acc.idxmax()]


# ------------------------------------------------------------- 1. tuning ---
TUNING = [  # file, title, hyperparameter on the x-axis, other hyperparameters (fixed at the scheme's best value)
    ("lr_tuning.csv", "Logistic regression", "C", []),
    ("svm_tuning.csv", "SVM (features)", "C", ["kernel"]),
    ("pca_tuning.csv", "SVM + PCA (images)", "C", ["n_components", "kernel"]),
    ("rf_tuning.csv", "Random forest", "max_depth", ["n_estimators"]),
]
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True, constrained_layout=True)
    for ax, (file, title, x, others) in zip(axes.flat, TUNING):
        t = pd.read_csv(OUT / file)
        fixed_text = []
        for scheme in ["LOSO", "Random"]:
            best = best_row(t, scheme)
            r = t[t.scheme == scheme]
            for o in others:
                r = r[r[o].astype(str) == str(best[o])]
            xs = [label(v) for v in r[x]]
            name = "LOSO" if scheme == "LOSO" else "Random split"
            ax.plot(xs, r.val_acc, "o-", color=COLORS[scheme], ms=3.5, label=f"{name}: validation")
            ax.plot(xs, r.train_acc, "--", color=COLORS[scheme], alpha=0.6, label=f"{name}: training")
            ax.scatter(label(best[x]), best.val_acc, s=70, facecolors="none", edgecolors="black", zorder=3,
                       label="Selected setting")
            if others:
                fixed = ", ".join(f"{o.replace('n_components', 'PCA').replace('n_estimators', 'trees')} {label(best[o])}"
                                  for o in others)
                fixed_text.append(f"{name}: {fixed}")
        ax.set_title(title + ("\n" + "; ".join(fixed_text) if fixed_text else "\n"), fontsize=8.5)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    keep = {l: h for h, l in zip(handles, labels_)}                                   # one entry per label
    fig.legend(keep.values(), keep.keys(), loc="outside upper center", ncol=5, fontsize=7, frameon=False)
    fig.savefig(OUT / "report_tuning.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / "report_tuning.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------- 2. comparison ---
comp = pd.read_csv(OUT / "comparison.csv")
cnn = pd.read_csv(OUT / "cnn_depth_search.csv").set_index("n_blocks").loc[5]
names = ["LR", "SVM (features)", "RF", "SVM + PCA (images)", "CNN (images)"]
loso = list(comp.set_index("model").loc[names[:4], "LOSO accuracy"]) + [cnn["LOSO validation"]]
loso_std = list(comp.set_index("model").loc[names[:4], "LOSO std"]) + [np.nan]
cv = list(comp.set_index("model").loc[names[:4], "Random CV accuracy"]) + [cnn["Random validation"]]
test = list(comp.set_index("model").loc[names[:4], "Test accuracy"]) + [np.nan]
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(len(names))
    bars = [("LOSO (new person)", loso, "C0"), ("Random split, validation (5-fold CV)", cv, "C1"),
            ("Random split, test set", test, "C2")]
    for k, (name, values, color) in enumerate(bars):
        pos = x + (k - 1) * 0.27
        err = loso_std if k == 0 else None
        ax.bar(pos, np.nan_to_num(values), width=0.25, color=color, label=name, yerr=err, capsize=2,
               error_kw={"elinewidth": 0.8})
        for p, v in zip(pos, values):
            if np.isnan(v):
                ax.text(p, 0.02, "not run", ha="center", va="bottom", fontsize=6, color="gray", rotation=90)
            else:
                ax.text(p, 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="white",
                        fontweight="bold", rotation=90)
    ax.axhline(0.25, color="gray", ls=":", lw=0.8)
    ax.text(len(names) - 0.5, 0.26, "chance", fontsize=6.5, color="gray", ha="right")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.savefig(OUT / "report_comparison.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / "report_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------- 3. per person + confusion ---
pp = pd.read_csv(OUT / "per_person_accuracy.csv").set_index("person")[names[:4]]
data = np.load(ROOT / "data" / "processed" / "wicos_dataset_jim.npz", allow_pickle=True)
X, y, persons, gestures = data["X_feat"], data["y"], data["persons"], list(data["gestures"])
best_C = float(comp.set_index("model").loc["LR", "LOSO params"].split("=")[1])     # best LOSO model: LR
lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=best_C, max_iter=10000))])
pred = cross_val_predict(lr, X, y, groups=persons, cv=LeaveOneGroupOut())
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.7, 1]})
    pp.plot.bar(ax=axes[0], width=0.8, rot=0, color=["C0", "C1", "C2", "C3"])
    axes[0].axhline(0.25, color="gray", ls=":", lw=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Held-out person")
    axes[0].set_ylabel("LOSO accuracy")
    axes[0].set_title("(a) Accuracy per held-out person", fontsize=9)
    axes[0].set_ylim(0, 1.17)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    axes[0].legend(fontsize=6.5, ncol=4, loc="upper center", frameon=False)
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=gestures, ax=axes[1], colorbar=False)
    axes[1].set_title(f"(b) LR, LOSO (C = {best_C:g})", fontsize=9)
    fig.savefig(OUT / "report_persons.pdf", bbox_inches="tight")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / "report_persons.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

for f in ["report_tuning", "report_comparison", "report_persons"]:
    print("saved", f + ".pdf")
