import numpy as np
import pandas as pd
import h5py
import json
import gc
import psutil
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 禁用可能的GPU使用（确保所有分类器使用CPU）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置线程数 - 根据您的128核CPU，我们可以设置合适的线程数
MAX_THREADS = 32  # 不要使用全部128核，避免内存竞争

# ============================
# Configuration
# ============================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def print_memory_usage(label=""):
    """打印内存使用情况"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 3
    print(f"{label} Memory usage: {mem:.2f} GB")


# 配置参数 - 进一步减少数据量确保稳定性
CONFIG = {
    "DrugCombDB": {
        "sampling_ratio": 0.2,
        "max_train": 100000,
        "max_val": 15000,
        "max_test": 15000,
    },
    "SynDrugMD": {
        "sampling_ratio": 0.15,
        "max_train": 100000,  # 进一步减少到10万
        "max_val": 15000,
        "max_test": 15000,
    }
}


# ============================
# Data Loader
# ============================
class DataLoader:
    def __init__(self, metadata_path, random_state=42):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.rng = np.random.RandomState(random_state)
        self.h5_cache = {}

    def _get_h5_file(self, file_path):
        if file_path not in self.h5_cache:
            self.h5_cache[file_path] = h5py.File(file_path, 'r')
        return self.h5_cache[file_path]

    def load_sampled_data(self, set_name, sampling_ratio=1.0, max_samples=None, random_state=42):
        features_list = []
        labels_list = []

        for data_type in ['positive', 'negative']:
            for batch_info in self.metadata[set_name][data_type]:
                h5_file = self._get_h5_file(batch_info['file'])
                group = h5_file[batch_info['group_path']]

                features = group['features'][:]
                labels = group['labels'][:]

                if sampling_ratio < 1.0:
                    n = len(features)
                    n_sample = int(n * sampling_ratio)
                    if n_sample > 0:
                        indices = self.rng.choice(n, n_sample, replace=False)
                        features = features[indices]
                        labels = labels[indices]
                    else:
                        continue

                features_list.append(features)
                labels_list.append(labels)

        if features_list:
            X = np.concatenate(features_list, axis=0)
            y = np.concatenate(labels_list, axis=0)

            indices = np.arange(len(X))
            self.rng.shuffle(indices)

            if max_samples and len(X) > max_samples:
                indices = indices[:max_samples]

            return X[indices], y[indices]
        return np.array([]), np.array([])

    def close(self):
        for h5_file in self.h5_cache.values():
            h5_file.close()


# ============================
# Classifier Benchmark - CPU Optimized
# ============================
class ClassifierBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.init_classifiers()
        self.results = {}

    def init_classifiers(self):
        # 所有分类器使用CPU，并限制线程数
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=MAX_THREADS,  # 限制线程数
                n_estimators=100,
                max_depth=15,
                max_samples=0.5,
                verbose=0
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=MAX_THREADS,  # 限制线程数
                verbose=0
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=15
            ),
            'Gaussian NB': GaussianNB(),
            'MLP': MLPClassifier(
                random_state=self.random_state,
                max_iter=200,
                hidden_layer_sizes=(128, 64),  # 更小的网络
                batch_size=256,
                early_stopping=True,
                n_iter_no_change=10,
                verbose=0,
                solver='adam',  # 使用adam而不是lbfgs
                alpha=0.0001,  # L2正则化
                learning_rate_init=0.001
            ),
            'XGBoost': None  # 先注释掉XGBoost，使用scikit-learn的分类器
        }

    def compute_metrics(self, y_true, y_pred, y_pred_proba):
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        metrics = {
            'Acc': accuracy_score(y_true, y_pred),
            'Sen': recall_score(y_true, y_pred),
            'Spe': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'Pre': precision_score(y_true, y_pred),
            'Mcc': matthews_corrcoef(y_true, y_pred),
            'AUROC': roc_auc_score(y_true, y_pred_proba),
            'AUPRC': average_precision_score(y_true, y_pred_proba),
            'F1': f1_score(y_true, y_pred)
        }
        return metrics

    def train_evaluate_classifier(self, classifier_name, classifier, X_train, y_train, X_val, y_val, X_test, y_test):
        print(f"  Training {classifier_name}...")
        classifier.fit(X_train, y_train)

        print(f"  Evaluating {classifier_name}...")
        y_pred = classifier.predict(X_test)

        # 处理概率预测
        if hasattr(classifier, 'predict_proba'):
            try:
                y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = y_pred.astype(float)
        else:
            y_pred_proba = y_pred.astype(float)

        metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        return {
            'metrics': metrics,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def run_benchmark(self, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
        print(f"Running benchmark on {dataset_name}...")
        print_memory_usage("Before training:")

        dataset_results = {}
        classifiers_list = list(self.classifiers.items())

        for i, (name, clf) in enumerate(classifiers_list):
            if clf is None:
                print(f"\n[{i + 1}/{len(classifiers_list)}] Skipping {name}")
                dataset_results[name] = None
                continue

            print(f"\n[{i + 1}/{len(classifiers_list)}] Processing {name}")

            try:
                result = self.train_evaluate_classifier(name, clf, X_train, y_train, X_val, y_val, X_test, y_test)
                dataset_results[name] = result
                print(f"  {name}: AUROC={result['metrics']['AUROC']:.4f}, AUPRC={result['metrics']['AUPRC']:.4f}")

                # 清理内存
                del clf
                gc.collect()

            except Exception as e:
                print(f"  {name} failed: {str(e)[:200]}")
                dataset_results[name] = None
                gc.collect()

        self.results[dataset_name] = dataset_results
        print_memory_usage("After training:")
        return dataset_results

    def plot_radar(self, dataset_name, save_path=None):
        if dataset_name not in self.results:
            return

        results = {k: v for k, v in self.results[dataset_name].items() if v is not None and k is not None}
        if not results:
            return

        metrics_names = ['Acc', 'Sen', 'Spe', 'Pre', 'Mcc', 'AUROC', 'AUPRC', 'F1']
        n_metrics = len(metrics_names)

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        for idx, (clf_name, result) in enumerate(results.items()):
            metrics = result['metrics']
            values = [metrics[name] for name in metrics_names]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=clf_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.2, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)

        plt.title(f'{dataset_name} - Performance Radar', size=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_curves(self, dataset_name, save_path=None):
        if dataset_name not in self.results:
            return

        results = {k: v for k, v in self.results[dataset_name].items() if v is not None and k is not None}
        if not results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        for idx, (clf_name, result) in enumerate(results.items()):
            fpr, tpr = result['roc_curve']
            auroc = result['metrics']['AUROC']
            ax1.plot(fpr, tpr, color=colors[idx], linewidth=2,
                     label=f'{clf_name} (AUC={auroc:.3f})')

            precision, recall = result['pr_curve']
            auprc = result['metrics']['AUPRC']
            ax2.plot(recall, precision, color=colors[idx], linewidth=2,
                     label=f'{clf_name} (AP={auprc:.3f})')

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{dataset_name} - ROC Curves')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        pos_ratio = np.mean([r['y_true'].mean() for r in results.values()])
        ax2.plot([0, 1], [pos_ratio, pos_ratio], 'k--', alpha=0.5,
                 label=f'Random (AP={pos_ratio:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{dataset_name} - PR Curves')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, output_dir="./benchmark_results"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, results in self.results.items():
            metrics_df = pd.DataFrame()
            for clf_name, result in results.items():
                if clf_name is not None and result is not None:
                    metrics_df[clf_name] = pd.Series(result['metrics'])

            if not metrics_df.empty:
                metrics_df.to_csv(f"{output_dir}/{dataset_name}_metrics.csv")
                print(f"Saved metrics: {output_dir}/{dataset_name}_metrics.csv")


# ============================
# Main Execution
# ============================
def run_experiment(dataset_name, metadata_path, output_dir="./benchmark_results"):
    config = CONFIG[dataset_name]

    loader = DataLoader(metadata_path, random_state=RANDOM_STATE)

    print(f"\nLoading {dataset_name} data...")
    X_train, y_train = loader.load_sampled_data(
        "train",
        sampling_ratio=config["sampling_ratio"],
        max_samples=config["max_train"],
        random_state=RANDOM_STATE
    )

    X_val, y_val = loader.load_sampled_data(
        "val",
        sampling_ratio=config["sampling_ratio"],
        max_samples=config["max_val"],
        random_state=RANDOM_STATE
    )

    X_test, y_test = loader.load_sampled_data(
        "test",
        sampling_ratio=config["sampling_ratio"],
        max_samples=config["max_test"],
        random_state=RANDOM_STATE
    )

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {X_train.shape} (pos: {y_train.sum()}/{len(y_train)})")
    print(f"  Val:   {X_val.shape} (pos: {y_val.sum()}/{len(y_val)})")
    print(f"  Test:  {X_test.shape} (pos: {y_test.sum()}/{len(y_test)})")

    # 释放不需要的数据
    gc.collect()
    print_memory_usage("After loading data:")

    benchmark = ClassifierBenchmark(random_state=RANDOM_STATE)
    benchmark.run_benchmark(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name)

    # 保存结果
    benchmark.save_results(output_dir)

    # 可视化
    try:
        benchmark.plot_radar(dataset_name, f"{output_dir}/{dataset_name}_radar.png")
        benchmark.plot_curves(dataset_name, f"{output_dir}/{dataset_name}_curves.png")
    except Exception as e:
        print(f"Visualization failed: {e}")

    loader.close()
    return benchmark


def main():
    import os
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "SynDrugMD": "./feature_batches/dataset_metadata.json"
    }

    all_results = {}

    # 先运行小数据集测试
    print("Testing with DrugCombDB first...")
    if "DrugCombDB" in datasets:
        benchmark = run_experiment("DrugCombDB", datasets["DrugCombDB"], output_dir)
        all_results["DrugCombDB"] = benchmark.results["DrugCombDB"]

    # 清理内存后再运行大数据集
    gc.collect()
    print("\n" + "=" * 60)
    print("Now testing with SynDrugMD...")
    print("=" * 60)

    if "SynDrugMD" in datasets:
        benchmark = run_experiment("SynDrugMD", datasets["SynDrugMD"], output_dir)
        all_results["SynDrugMD"] = benchmark.results["SynDrugMD"]

    generate_summary_report(all_results, output_dir)


def generate_summary_report(all_results, output_dir):
    print(f"\n{'=' * 60}")
    print("SUMMARY REPORT")
    print(f"{'=' * 60}")

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)

        summary_data = []
        for clf_name, result in results.items():
            if clf_name is not None and result is not None:
                metrics = result['metrics']
                summary_data.append({
                    'Classifier': clf_name,
                    'Acc': f"{metrics['Acc']:.4f}",
                    'Sen': f"{metrics['Sen']:.4f}",
                    'Spe': f"{metrics['Spe']:.4f}",
                    'Pre': f"{metrics['Pre']:.4f}",
                    'Mcc': f"{metrics['Mcc']:.4f}",
                    'AUROC': f"{metrics['AUROC']:.4f}",
                    'AUPRC': f"{metrics['AUPRC']:.4f}",
                    'F1': f"{metrics['F1']:.4f}"
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))

            auroc_vals = df['AUROC'].astype(float)
            best_idx = auroc_vals.idxmax()
            best_clf = df.iloc[best_idx]
            print(f"\nBest classifier (AUROC): {best_clf['Classifier']} (AUROC={best_clf['AUROC']})")


if __name__ == "__main__":
    print("Starting classifier benchmark...")
    print(f"Using up to {MAX_THREADS} CPU threads")
    main()