"""Generate publication-ready curves for the GPU-lite ablation results."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "output" / "results_gpu_lite"
REPORTS = ROOT / "output" / "reports_gpu_lite"
METHODS = {"mean_euclidean": "Mean Euclidean", "rms": "RMS"}
THRESHOLDS = (0.9, 1.0, 1.1)
SEEDS = (42, 52, 62, 72, 82)
COLORS = {"mean_euclidean": "#2374AB", "rms": "#D1495B"}
LINESTYLES = {0.9: "-", 1.0: "--", 1.1: ":"}


def setup_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.unicode_minus": False,
    })


def group_dir(method: str, threshold: float) -> Path:
    return RESULTS / f"{method}_aed_{threshold:.1f}"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS / f"{name}.png", bbox_inches="tight")
    fig.savefig(REPORTS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_thresholds() -> None:
    data = pd.read_csv(REPORTS / "all_seed_metrics.csv")
    data = data[data["split"] == "test"]
    metrics = [("acc", "ACC"), ("sen", "SEN"), ("f1", "F1"),
               ("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("aupr", "AUPR")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method, method_label in METHODS.items():
            part = data[data["method"] == method].groupby("threshold")[metric].agg(["mean", "std"])
            x = part.index.to_numpy()
            ax.errorbar(x, part["mean"], yerr=part["std"], marker="o", capsize=4,
                        linewidth=2, color=COLORS[method], label=method_label)
        ax.set_title(label)
        ax.set_xlabel("AED 阈值倍数")
        ax.set_ylabel("指标值")
        ax.set_xticks(THRESHOLDS)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("不同负样本筛选阈值的测试集性能（均值 ± 标准差）", fontsize=15)
    save(fig, "01_阈值-性能曲线")


def mean_curve(method: str, threshold: float, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0, 1, 501)
    curves = []
    for seed in SEEDS:
        pred = pd.read_csv(group_dir(method, threshold) / f"seed_{seed}" / "test_pair_predictions.csv")
        y, score = pred["label"].to_numpy(), pred["probability"].to_numpy()
        if kind == "roc":
            x, values, _ = roc_curve(y, score)
            curves.append(np.interp(grid, x, values))
        else:
            precision, recall, _ = precision_recall_curve(y, score)
            order = np.argsort(recall)
            curves.append(np.interp(grid, recall[order], precision[order]))
    matrix = np.vstack(curves)
    return grid, matrix.mean(axis=0), matrix.std(axis=0)


def plot_roc_pr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    for method, method_label in METHODS.items():
        for threshold in THRESHOLDS:
            x, mean, std = mean_curve(method, threshold, "roc")
            label = f"{method_label}, {threshold:.1f}×AED"
            axes[0].plot(x, mean, color=COLORS[method], linestyle=LINESTYLES[threshold],
                         linewidth=2, label=label)
            if threshold == 0.9:
                axes[0].fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                     color=COLORS[method], alpha=0.10)
            x, mean, std = mean_curve(method, threshold, "pr")
            axes[1].plot(x, mean, color=COLORS[method], linestyle=LINESTYLES[threshold],
                         linewidth=2, label=label)
            if threshold == 0.9:
                axes[1].fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                     color=COLORS[method], alpha=0.10)
    axes[0].plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
    axes[0].set(title="测试集 ROC 曲线", xlabel="假阳性率（FPR）", ylabel="真阳性率（TPR）")
    axes[1].set(title="测试集 PR 曲线", xlabel="召回率（Recall）", ylabel="精确率（Precision）")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("两种负样本距离方法的 ROC / PR 曲线（5 个种子平均）", fontsize=15)
    save(fig, "02_ROC-PR曲线")


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    for method, method_label in METHODS.items():
        for threshold in THRESHOLDS:
            histories = []
            for seed in SEEDS:
                histories.append(pd.read_csv(group_dir(method, threshold) / f"seed_{seed}" / "training_history.csv"))
            for metric, ax in (("logloss", axes[0]), ("auc", axes[1])):
                matrix = np.vstack([h[metric].to_numpy() for h in histories])
                x = np.arange(1, matrix.shape[1] + 1)
                mean, std = matrix.mean(axis=0), matrix.std(axis=0)
                label = f"{method_label}, {threshold:.1f}×AED"
                ax.plot(x, mean, color=COLORS[method], linestyle=LINESTYLES[threshold],
                        linewidth=1.8, label=label)
                if threshold == 0.9:
                    ax.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.08)
    axes[0].set(title="训练集 Logloss", xlabel="Boosting 轮数", ylabel="Logloss")
    axes[1].set(title="训练集 AUC", xlabel="Boosting 轮数", ylabel="AUC")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("XGBoost 训练曲线（5 个种子平均）", fontsize=15)
    save(fig, "03_训练曲线")


def plot_reference_three_panel() -> None:
    """Plot only Loss, ROC and PR/AUPR for the best 0.9xAED setting."""
    threshold = 0.9
    fig = plt.figure(figsize=(13.5, 7.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), hspace=0.34, wspace=0.28)
    loss_ax = fig.add_subplot(grid[:, 0])
    roc_ax = fig.add_subplot(grid[0, 1])
    pr_ax = fig.add_subplot(grid[1, 1])

    for method, method_label in METHODS.items():
        histories = [
            pd.read_csv(group_dir(method, threshold) / f"seed_{seed}" / "training_history.csv")
            for seed in SEEDS
        ]
        matrix = np.vstack([history["logloss"].to_numpy() for history in histories])
        rounds = np.arange(1, matrix.shape[1] + 1)
        mean, std = matrix.mean(axis=0), matrix.std(axis=0)
        loss_ax.plot(rounds, mean, color=COLORS[method], linewidth=2, label=method_label)
        loss_ax.fill_between(rounds, mean - std, mean + std, color=COLORS[method], alpha=0.12)

        x, mean, std = mean_curve(method, threshold, "roc")
        roc_ax.plot(x, mean, color=COLORS[method], linewidth=2, label=method_label)
        roc_ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=COLORS[method], alpha=0.12)

        x, mean, std = mean_curve(method, threshold, "pr")
        pr_ax.plot(x, mean, color=COLORS[method], linewidth=2, label=method_label)
        pr_ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                           color=COLORS[method], alpha=0.12)

    loss_ax.set(title="A  Loss", xlabel="Boosting 轮数", ylabel="训练集 Logloss")
    loss_ax.legend(frameon=False, loc="upper right")
    roc_ax.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
    roc_ax.set(title="B  ROC", xlabel="假阳性率（FPR）", ylabel="真阳性率（TPR）",
               xlim=(0, 1), ylim=(0, 1.02))
    pr_ax.set(title="C  AUPR（PR 曲线）", xlabel="召回率（Recall）", ylabel="精确率（Precision）",
              xlim=(0, 1), ylim=(0, 1.02))
    pr_ax.legend(frameon=False, loc="lower left")
    fig.suptitle("0.9×AED 模型性能曲线（5 个随机种子均值 ± 标准差）", fontsize=15)
    save(fig, "04_ROC-AUPR-Loss三联图")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_metric_thresholds()
    plot_roc_pr()
    plot_training_curves()
    plot_reference_three_panel()
    print(f"曲线图已保存到：{REPORTS}")


if __name__ == "__main__":
    main()
