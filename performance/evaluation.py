import gc
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

import config

matplotlib.use('Agg')

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

METRIC_ORDER = ['Acc', 'Sen', 'Spe', 'Pre', 'Mcc', 'AUROC', 'AUPRC', 'F1']
DATASET_NAME = 'SynDrugMD'

PLOT_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#e377c2', '#bcbd22', '#17becf']
METRIC_LABELS = [('Accuracy', 'Acc'), ('Sensitivity', 'Sen'), ('Specificity', 'Spe'),
                 ('Precision', 'Pre'), ('MCC', 'Mcc'), ('AUROC', 'AUROC'),
                 ('AUPR', 'AUPRC'), ('F1_Score', 'F1')]


def compute_pair_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return {
        'Acc': float(accuracy_score(y_true, y_pred)),
        'Sen': float(recall_score(y_true, y_pred, zero_division=0)),
        'Spe': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'Pre': float(precision_score(y_true, y_pred, zero_division=0)),
        'Mcc': float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        'AUROC': float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
        'AUPRC': float(average_precision_score(y_true, y_score)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_benchmark(data, cfg):
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    results = {}

    from classifiers import init_classifiers, predict_proba, train

    for name, clf in init_classifiers(config.RANDOM_STATE).items():
        try:
            train(clf, data['X_train'], data['y_train'])
            y_val_score = predict_proba(clf, data['X_val'])
            y_test_score = predict_proba(clf, data['X_test'])
            m_val = compute_pair_metrics(data['y_val'], y_val_score, cfg['threshold'])
            m_test = compute_pair_metrics(data['y_test'], y_test_score, cfg['threshold'])
            fpr, tpr, _ = roc_curve(data['y_test'], y_test_score)
            prec, rec, _ = precision_recall_curve(data['y_test'], y_test_score)
            results[name] = {
                'metrics_val': m_val,
                'metrics': m_test,
                'roc_curve': (fpr, tpr),
                'pr_curve': (prec, rec),
            }
        except Exception:
            pass
        finally:
            gc.collect()

    _export_reference_metrics(results)
    _plot_reference_figure(results, Path(config.OUTPUT_DIR) / f'{DATASET_NAME}_figure.png')


def _export_reference_metrics(results):
    metrics_df = pd.DataFrame()
    for clf_name, result in results.items():
        if result is not None:
            metrics_df[clf_name] = pd.Series(result['metrics'])
    if not metrics_df.empty:
        metrics_df.to_csv(Path(config.OUTPUT_DIR) / f'{DATASET_NAME}_metrics.csv')


def _plot_reference_figure(results, save_path):
    if not results:
        return
    clf_names = list(results)
    colors = {name: PLOT_COLORS[i % len(PLOT_COLORS)] for i, name in enumerate(clf_names)}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.17, wspace=0.22)

    n_groups = len(METRIC_LABELS)
    bar_width = 0.13
    x = np.arange(n_groups)
    for i, name in enumerate(clf_names):
        values = [results[name]['metrics'][key] for _, key in METRIC_LABELS]
        ax_a.bar(x + i * bar_width, values, width=bar_width,
                 label=name, color=colors[name], edgecolor='none')
    centers = x + bar_width * (len(clf_names) - 1) / 2
    ax_a.set_xticks(centers)
    ax_a.set_xticklabels([label for label, _ in METRIC_LABELS], fontsize=9, rotation=20,
                          ha='right')
    ax_a.set_ylim(0.5, 1.0)
    ax_a.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax_a.tick_params(axis='y', labelsize=9)
    ax_a.grid(True, linestyle=':', color='lightgray', linewidth=0.8, alpha=0.9)
    ax_a.set_axisbelow(True)
    ax_a.set_box_aspect(1)
    ax_a.text(-0.08, 1.04, 'A', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='bottom')

    for name in clf_names:
        result = results[name]
        fpr, tpr = result['roc_curve']
        precision, recall = result['pr_curve']
        order = np.argsort(recall)
        rec_s = recall[order]
        prec_s = np.maximum.accumulate(precision[order][::-1])[::-1]
        ax_b.plot(fpr, tpr, color=colors[name], linewidth=1.4)
        ax_b.plot(rec_s, prec_s, color=colors[name], linewidth=1.4)

    ax_b.set_xlabel('False positive rate / Recall', fontsize=11)
    ax_b.set_ylabel('True positive rate / Precision', fontsize=11)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax_b.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_b.tick_params(labelsize=9)
    ax_b.grid(True, linestyle=':', color='lightgray', linewidth=0.8, alpha=0.9)
    ax_b.set_axisbelow(True)
    ax_b.set_box_aspect(1)
    ax_b.text(-0.08, 1.04, 'B', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='bottom')
    ax_b.text(0.36, 0.86, 'ROC', fontsize=11, style='normal')
    ax_b.text(0.82, 0.86, 'PRC', fontsize=11, style='normal')

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(clf_names),
               frameon=False, fontsize=9, columnspacing=1.6, handlelength=1.4,
               bbox_to_anchor=(0.5, 0.015))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
