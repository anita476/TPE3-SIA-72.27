"""
visualize_separability.py
=========================
Produces four plots that together make it visually clear that the fraud
dataset is NOT linearly separable.

Usage:
    python visualize_separability.py --data ../data/fraud_dataset.csv \
                                     --label flagged_fraud \
                                     --out results/plots

Plots produced:
    1. pca_2d.png          — PCA projection: can a line separate the classes?
    2. feature_dists.png   — Per-feature distributions: are classes cleanly split on any single feature?
    3. linear_boundary.png — Best linear boundary (logistic regression) with misclassified points highlighted
    4. pca_variance.png    — Cumulative PCA variance: how many dimensions do you actually need?
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "medium",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#dddddd",
})

COLOR_LEGIT  = "#378ADD"   # blue  — legitimate
COLOR_FRAUD  = "#D85A30"   # red   — fraud
COLOR_WRONG  = "#FFD700"   # gold  — misclassified


# ── data loading ──────────────────────────────────────────────────────────────

def load(path: str, label: str):
    df = pd.read_csv(path)
    y  = df[label].values.astype(int)
    X_pre = df.drop(columns=[label, "big_model_fraud_probability"])
    X  = X_pre.select_dtypes(include=[np.number]).values.astype(float)
    features = list(X_pre.select_dtypes(include=[np.number]).columns)
    return X, y, features

# ── plot 1: PCA 2-D scatter ───────────────────────────────────────────────────

def plot_pca_2d(X_scaled, y, out_dir: Path):
    pca  = PCA(n_components=2, random_state=42)
    X2   = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color, label in [(0, COLOR_LEGIT, "Legitimate"), (1, COLOR_FRAUD, "Fraud")]:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=color, alpha=0.35, s=12, linewidths=0, label=label)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title("PCA projection (2D) — can a straight line separate the classes?")
    ax.legend()

    # draw a reference horizontal and vertical line to make the question concrete
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    p = out_dir / "pca_2d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# ── plot 2: per-feature class distributions ───────────────────────────────────

def plot_feature_dists(X_scaled, y, features, out_dir: Path):
    n     = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax   = axes[i]
        vals = X_scaled[:, i]
        lo   = np.percentile(vals, 1)
        hi   = np.percentile(vals, 99)
        bins = np.linspace(lo, hi, 50)

        for cls, color, label in [(0, COLOR_LEGIT, "Legitimate"), (1, COLOR_FRAUD, "Fraud")]:
            ax.hist(vals[y == cls], bins=bins, color=color,
                    alpha=0.55, density=True, label=label)

        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Standardised value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

        # if distributions strongly overlap → not separable on this feature
        overlap = _overlap_score(vals[y == 0], vals[y == 1])
        ax.text(0.97, 0.95, f"overlap: {overlap:.0%}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=7, color="#555555")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color=COLOR_LEGIT, alpha=0.6, label="Legitimate"),
        mpatches.Patch(color=COLOR_FRAUD, alpha=0.6, label="Fraud"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Per-feature distributions — overlapping classes → not linearly separable",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = out_dir / "feature_dists.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


def _overlap_score(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    """Histogram intersection: 1.0 = complete overlap, 0.0 = fully separate."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width  = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb)) * width)


# ── plot 3: best linear boundary + misclassified points ──────────────────────

def plot_linear_boundary(X_scaled, y, out_dir: Path):
    """
    Fit the best possible linear classifier (full logistic regression, not a
    single perceptron) in PCA-2D space, then show the decision boundary and
    highlight every misclassified point.  If even the best linear model
    misclassifies heavily, the problem is not linearly separable.
    """
    pca  = PCA(n_components=2, random_state=42)
    X2   = pca.fit_transform(X_scaled)

    clf  = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(X2, y)
    pred = clf.predict(X2)
    acc  = accuracy_score(y, pred)
    wrong = pred != y

    # decision boundary grid
    h   = 0.05
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))

    # probability heatmap
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.25, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="#333333", linewidths=1.5,
               linestyles="--")

    # correctly classified
    for cls, color, label in [(0, COLOR_LEGIT, "Legitimate"), (1, COLOR_FRAUD, "Fraud")]:
        mask = (y == cls) & ~wrong
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=color, alpha=0.4, s=12, linewidths=0, label=label)

    # misclassified — shown in gold with a ring
    ax.scatter(X2[wrong, 0], X2[wrong, 1],
               facecolors="none", edgecolors=COLOR_WRONG,
               s=28, linewidths=0.8, label=f"Misclassified ({wrong.sum():,})")

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title(
        f"Best possible linear boundary in PCA-2D space\n"
        f"Accuracy = {acc:.1%}  |  Misclassified = {wrong.sum():,} / {len(y):,}  "
        f"({wrong.mean():.1%})"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    p = out_dir / "linear_boundary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}  (linear accuracy in 2D: {acc:.1%})")


# ── plot 4: PCA cumulative variance ──────────────────────────────────────────

def plot_pca_variance(X_scaled, out_dir: Path):
    pca  = PCA(random_state=42)
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_comp = np.arange(1, len(cumvar) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_comp, cumvar, color=COLOR_LEGIT, linewidth=2, marker="o", markersize=5)

    for threshold in [80, 90, 95]:
        n = int(np.searchsorted(cumvar, threshold)) + 1
        ax.axhline(threshold, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.text(n_comp[-1] * 0.98, threshold + 0.8, f"{threshold}% @ {n} PCs",
                ha="right", fontsize=8, color="#555555")

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA cumulative variance — how many dimensions encode the structure?")
    ax.set_xlim(1, len(cumvar))
    ax.set_ylim(0, 102)

    fig.tight_layout()
    p = out_dir / "pca_variance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise linear non-separability of a binary classification dataset."
    )
    parser.add_argument("--data",  required=True, help="Path to CSV dataset")
    parser.add_argument("--label", required=True, help="Name of the binary label column")
    parser.add_argument("--out",   default="results/plots", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.data}")
    X, y, features = load(args.data, args.label)
    print(f"  {X.shape[0]:,} samples | {X.shape[1]} features | "
          f"fraud rate={y.mean()*100:.2f}%\n")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Generating plots...")
    plot_pca_2d(X_scaled, y, out_dir)
    plot_feature_dists(X_scaled, y, features, out_dir)
    plot_linear_boundary(X_scaled, y, out_dir)
    plot_pca_variance(X_scaled, out_dir)

    print(f"\nDone. All plots saved to: {out_dir}")
    print("""
What each plot tells you:
  pca_2d.png          — If classes are interleaved/mixed in 2D, no line can separate them
  feature_dists.png   — High overlap scores on every feature = no single feature separates classes
  linear_boundary.png — Even the best possible linear model misclassifies many points
  pca_variance.png    — If many PCs are needed, the structure lives in high-dimensional space
                        that a single linear boundary cannot carve up
""")


if __name__ == "__main__":
    main()