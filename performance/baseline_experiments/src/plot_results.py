from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = ("#2878B5", "#D95319", "#E5AE00", "#4C9F70", "#8E44AD")
GRID = 400
FONT_SIZE = 18

PANEL_A_ORDER: tuple[str, ...] = ("ACC", "SEN", "SPE", "PRE", "MCC", "AUROC", "AUPRC", "F1")
SHORT_LABELS: dict[str, str] = {
    "KGNN": "KGNN", "GraRep": "GraRep", "DeepDDS": "DeepDDS (adapted)",
    "DeepTraSynergy": "DeepTraSynergy (adapted)", "KGP-DC": "KGP-DC",
}

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "axes.grid": False,
})

def load_curves(model_key: str, folds: list[int], seed: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, GRID)
    roc, prc = [], []
    for fold in folds:
        if not common.prediction_path(model_key, fold, seed).exists():
            continue
        frame = common.load_predictions(model_key, fold, seed)
        labels = frame["true_label"].to_numpy()
        probabilities = frame["predicted_probability"].to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        roc.append(np.interp(grid, fpr, tpr))
        order = np.argsort(recall)
        prc.append(np.interp(grid, recall[order], precision[order]))
    if not roc:
        raise FileNotFoundError(f"no predictions for {model_key}")
    return grid, np.vstack(roc).mean(axis=0), np.vstack(prc).mean(axis=0)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", nargs="+", type=int, default=list(common.FOLDS))
    parser.add_argument("--seed", type=int, default=common.SEEDS[0])
    args = parser.parse_args()

    if not common.TABLE_NUMERIC_CSV.exists():
        raise SystemExit(f"missing {common.TABLE_NUMERIC_CSV}; run `python main.py evaluate`")
    table = pd.read_csv(common.TABLE_NUMERIC_CSV)
    model_keys = [key for key in common.MODEL_KEYS if key in set(table["model_key"])]

    figure, axes = plt.subplots(
        1, 2, figsize=(24.0, 7.6), gridspec_kw={"width_ratios": [1.8, 1.0]})

    axis = axes[0]
    positions = np.arange(len(PANEL_A_ORDER))
    width = 0.82 / max(1, len(model_keys))
    lower = 1.0
    for index, model_key in enumerate(model_keys):
        row = table[table["model_key"] == model_key].iloc[0]
        values = [float(row[f"{metric}_mean"]) for metric in PANEL_A_ORDER]
        lower = min(lower, min(values))
        axis.bar(positions + (index - (len(model_keys) - 1) / 2) * width, values,
                 width * 0.9, color=COLORS[index % len(COLORS)],
                 edgecolor="black", linewidth=0.4, label=SHORT_LABELS[model_key])
    axis.set_xticks(positions)
    axis.set_xticklabels([common.METRIC_LABELS[metric] for metric in PANEL_A_ORDER])
    axis.set_ylim(max(0.0, np.floor((lower - 0.12) * 10) / 10), 1.0)
    axis.set_ylabel("Score")
    axis.set_title("A", loc="left", fontsize=FONT_SIZE + 4, fontweight="bold")
    axis.grid(False)

    axis = axes[1]
    for index, model_key in enumerate(model_keys):
        color = COLORS[index % len(COLORS)]
        grid, roc_mean, prc_mean = load_curves(model_key, args.folds, args.seed)
        axis.plot(grid, roc_mean, color=color, linewidth=2.2, linestyle="-")
        axis.plot(grid, prc_mean, color=color, linewidth=1.8, linestyle="-")
    axis.text(0.26, 0.72, "ROC", fontsize=FONT_SIZE + 2, fontweight="bold")
    axis.text(0.62, 0.84, "PRC", fontsize=FONT_SIZE + 2, fontweight="bold")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: "0" if abs(value) < 1e-9 else f"{value:.1f}"))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: "" if abs(value) < 1e-9 else f"{value:.1f}"))
    axis.set_xlabel("False positive rate / Recall")
    axis.set_ylabel("True positive rate / Precision")
    axis.set_title("B", loc="left", fontsize=FONT_SIZE + 4, fontweight="bold")
    axis.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[index % len(COLORS)],
                             edgecolor="black", linewidth=0.4)
               for index in range(len(model_keys))]
    figure.legend(handles, [SHORT_LABELS[key] for key in model_keys],
                  loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=5,
                  frameon=False)
    figure.tight_layout()
    figure.subplots_adjust(wspace=0.45)
    figure.savefig(common.FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[figure] {common.FIGURE_PATH}")

if __name__ == "__main__":
    main()
