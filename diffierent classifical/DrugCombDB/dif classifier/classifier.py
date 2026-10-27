import numpy as np
import pandas as pd
import h5py
import json
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import gc

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)

# 设置字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 减少数据量，避免内存问题
CONFIG = {
    "DrugCombDB": {
        "sampling_ratio": 0.2,
        "max_train": 50000,
        "max_val": 10000,
        "max_test": 10000,
    },
    "SynDrugMD": {
        "sampling_ratio": 0.15,
        "max_train": 80000,
        "max_val": 15000,
        "max_test": 15000,
    }
}


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

    def load_sampled_data(self, set_name, sampling_ratio=1.0, max_samples=None,
                          add_noise=False, noise_level=0.05,
                          feature_shuffle=False, shuffle_prob=0.3,
                          feature_dropout=False, dropout_prob=0.2):

        features_list = []
        labels_list = []

        for data_type in ['positive', 'negative']:
            if set_name in self.metadata and data_type in self.metadata[set_name]:
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

                    n_samples = len(features)

                    if add_noise and noise_level > 0:
                        noise = self.rng.normal(0, noise_level, features.shape)
                        features = features + noise

                        salt_pepper_prob = 0.1
                        salt_pepper_mask = self.rng.random(features.shape) < salt_pepper_prob
                        salt_values = self.rng.choice([-1, 1], size=features.shape, p=[0.5, 0.5])
                        features = np.where(salt_pepper_mask, salt_values * noise_level * 5, features)

                    if feature_shuffle and shuffle_prob > 0:
                        n_shuffle = int(n_samples * shuffle_prob)
                        if n_shuffle > 0:
                            shuffle_indices = self.rng.choice(n_samples, n_shuffle, replace=False)
                            for idx in shuffle_indices:
                                shuffle_idx = self.rng.permutation(features.shape[1])
                                features[idx] = features[idx][shuffle_idx]

                    if feature_dropout and dropout_prob > 0:
                        dropout_mask = self.rng.random(features.shape) > dropout_prob
                        features = features * dropout_mask

                    features_list.append(features)
                    labels_list.append(labels)

                    # 及时清理内存
                    gc.collect()

        if features_list:
            X = np.concatenate(features_list, axis=0)
            y = np.concatenate(labels_list, axis=0)

            indices = self.rng.permutation(len(X))
            X = X[indices]
            y = y[indices]

            if max_samples and len(X) > max_samples:
                indices = self.rng.choice(len(X), max_samples, replace=False)
                X = X[indices]
                y = y[indices]

            return X, y
        return np.array([]), np.array([])

    def close(self):
        for h5_file in self.h5_cache.values():
            h5_file.close()


class ClassifierBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifiers = self._init_classifiers()
        self.results = {}
        self.curve_data = {}

    def _init_classifiers(self):
        # 关键修改：在 Windows 上减少并行线程数，避免内存冲突
        # 使用单线程或少量线程
        classifiers = {
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=2,  # 减少到2线程
                n_estimators=100,  # 减少树的数量
                max_depth=15,
                max_samples=0.5,
                verbose=0
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=500,  # 减少迭代次数
                n_jobs=1,  # Windows 上使用单线程
                solver='lbfgs',  # 使用更稳定的求解器
                verbose=0
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=15
            ),
            'Gaussian NB': GaussianNB(),
            'MLP': MLPClassifier(
                random_state=self.random_state,
                max_iter=300,
                hidden_layer_sizes=(100, 50),
                early_stopping=True,
                solver='adam',
                verbose=0
            )
        }

        try:
            from xgboost import XGBClassifier
            classifiers['XGBoost'] = XGBClassifier(
                random_state=self.random_state,
                n_jobs=2,  # 减少到2线程
                n_estimators=100,
                max_depth=8,
                tree_method='hist',  # 使用直方图方法，内存更友好
                verbosity=0,
                use_label_encoder=False
            )
        except ImportError:
            pass

        return classifiers

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

    def train_evaluate_single(self, clf_name, clf, X_train, y_train, X_test, y_test):
        """训练和评估单个分类器，增加内存清理"""
        try:
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            if hasattr(clf, 'predict_proba'):
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred.astype(float)

            metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)

            # 计算曲线数据
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

            return metrics, fpr, tpr, precision, recall

        except Exception as e:
            print(f"Detailed error in {clf_name}: {str(e)}")
            raise

    def plot_roc_prc_curves(self, dataset_name, output_dir="./benchmark_results"):
        """绘制ROC和PRC曲线在同一张图中"""
        if dataset_name not in self.curve_data:
            print(f"No curve data found for {dataset_name}")
            return

        curve_data = self.curve_data[dataset_name]

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 定义颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, len(curve_data)))

        # 准备图例
        legend_handles = []
        legend_labels = []

        # 绘制ROC和PRC曲线
        for i, (clf_name, data) in enumerate(curve_data.items()):
            if data is not None:
                # 获取ROC和PRC数据
                fpr, tpr, auroc = data['roc']
                precision, recall, auprc = data['prc']

                # 使用相同的颜色和线型绘制ROC曲线
                line1, = plt.plot(fpr, tpr, color=colors[i], linewidth=12,
                                  solid_capstyle='round', solid_joinstyle='round')

                # 使用相同的颜色和线型绘制PRC曲线
                line2, = plt.plot(recall, precision, color=colors[i], linewidth=12,
                                  solid_capstyle='round', solid_joinstyle='round')

                # 添加图例项
                legend_handles.append(line1)
                legend_labels.append(f"{clf_name}")

        # 设置坐标轴
        plt.xlabel('False positive rate/Recall', fontsize=16, fontweight='bold')
        plt.ylabel('The positive rate/Precision', fontsize=16, fontweight='bold')

        # 设置坐标轴范围
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])

        # 设置坐标轴刻度
        plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)

        # 添加网格
        plt.grid(True, which='both', alpha=0.2, linestyle='-', linewidth=0.5)

        # 设置图例 - 位于左侧
        plt.legend(legend_handles, legend_labels,
                   loc='upper left', fontsize=12, frameon=True,
                   framealpha=0.9, edgecolor='black')

        # 移除标题
        plt.tight_layout()

        # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/{dataset_name}_ROC_PRC_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC and PRC curves saved to: {save_path}")

        # 保存曲线数据
        self.save_curve_data(dataset_name, output_dir)

    def save_curve_data(self, dataset_name, output_dir):
        """保存曲线数据到CSV文件"""
        if dataset_name not in self.curve_data:
            return

        curve_data = self.curve_data[dataset_name]

        curve_points = []

        for clf_name, data in curve_data.items():
            if data is not None:
                # ROC曲线点
                fpr, tpr, auroc = data['roc']
                for fpr_val, tpr_val in zip(fpr, tpr):
                    curve_points.append({
                        'Dataset': dataset_name,
                        'Classifier': clf_name,
                        'Curve_Type': 'ROC',
                        'X': fpr_val,
                        'Y': tpr_val,
                        'AUC': auroc
                    })

                # PRC曲线点
                precision, recall, auprc = data['prc']
                for rec_val, prec_val in zip(recall, precision):
                    curve_points.append({
                        'Dataset': dataset_name,
                        'Classifier': clf_name,
                        'Curve_Type': 'PRC',
                        'X': rec_val,
                        'Y': prec_val,
                        'AUC': auprc
                    })

        if curve_points:
            df = pd.DataFrame(curve_points)
            csv_path = f"{output_dir}/{dataset_name}_curve_points.csv"
            df.to_csv(csv_path, index=False)
            print(f"Curve points saved to: {csv_path}")

    def run_benchmark(self, X_train, y_train, X_test, y_test, dataset_name):
        dataset_results = {}
        dataset_curve_data = {}

        print(f"\n{'=' * 50}")
        print(f"Training classifiers for {dataset_name}")
        print(f"{'=' * 50}")

        for name, clf in self.classifiers.items():
            try:
                print(f"Training {name}...")
                metrics, fpr, tpr, precision, recall = self.train_evaluate_single(
                    name, clf, X_train, y_train, X_test, y_test
                )
                dataset_results[name] = metrics

                # 存储曲线数据
                dataset_curve_data[name] = {
                    'roc': (fpr, tpr, metrics['AUROC']),
                    'prc': (precision, recall, metrics['AUPRC'])
                }

                print(f"{name:20s} | AUROC: {metrics['AUROC']:.4f} | AUPRC: {metrics['AUPRC']:.4f} | "
                      f"Acc: {metrics['Acc']:.4f} | Sen: {metrics['Sen']:.4f} | Spe: {metrics['Spe']:.4f}")

                # 及时清理内存
                del clf
                gc.collect()

            except Exception as e:
                print(f"Error training {name}: {e}")
                dataset_results[name] = None
                dataset_curve_data[name] = None

        self.results[dataset_name] = dataset_results
        self.curve_data[dataset_name] = dataset_curve_data

        # 绘制该数据集的ROC和PRC曲线
        self.plot_roc_prc_curves(dataset_name)

        return dataset_results

    def save_results(self, output_dir="./benchmark_results"):
        os.makedirs(output_dir, exist_ok=True)

        performance_data = []

        for dataset_name, results in self.results.items():
            for clf_name, metrics in results.items():
                if metrics is not None:
                    performance_data.append({
                        'Dataset': dataset_name,
                        'Classifier': clf_name,
                        'Accuracy': metrics['Acc'],
                        'Sensitivity': metrics['Sen'],
                        'Specificity': metrics['Spe'],
                        'Precision': metrics['Pre'],
                        'MCC': metrics['Mcc'],
                        'AUROC': metrics['AUROC'],
                        'AUPRC': metrics['AUPRC'],
                        'F1_Score': metrics['F1']
                    })

        if performance_data:
            df = pd.DataFrame(performance_data)
            csv_path = f"{output_dir}/classifier_performance.csv"
            df.to_csv(csv_path, index=False)
            print(f"Performance results saved to: {csv_path}")


def run_experiment(dataset_name, metadata_path, output_dir="./benchmark_results"):
    config = CONFIG[dataset_name]

    if not os.path.exists(metadata_path):
        print(f"File not found: {metadata_path}")
        return None

    loader = DataLoader(metadata_path, random_state=RANDOM_STATE)

    X_train, y_train = loader.load_sampled_data(
        "train",
        sampling_ratio=config["sampling_ratio"],
        max_samples=config["max_train"],
        add_noise=True,
        noise_level=0.05,
        feature_shuffle=True,
        shuffle_prob=0.3,
        feature_dropout=True,
        dropout_prob=0.2
    )

    X_test, y_test = loader.load_sampled_data(
        "test",
        sampling_ratio=config["sampling_ratio"],
        max_samples=config["max_test"],
        add_noise=False,
        feature_shuffle=False,
        feature_dropout=False
    )

    print(f"\n{dataset_name}:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive samples: {np.sum(y_train == 1)}")
    print(f"  Negative samples: {np.sum(y_train == 0)}")

    benchmark = ClassifierBenchmark(random_state=RANDOM_STATE)
    benchmark.run_benchmark(X_train, y_train, X_test, y_test, dataset_name)

    loader.close()
    return benchmark


def main():
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    # 测试用的小数据集
    datasets = {
        "DrugCombDB": "./feature_batches/dataset_metadata.json",
    }

    all_results = {}

    for dataset_name, metadata_path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        benchmark = run_experiment(dataset_name, metadata_path, output_dir)
        if benchmark:
            all_results[dataset_name] = benchmark.results[dataset_name]

    # 保存所有结果
    save_combined_results(all_results, output_dir)

    print(f"\n{'=' * 60}")
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'=' * 60}")


def save_combined_results(all_results, output_dir):
    combined_data = []

    for dataset_name, results in all_results.items():
        for clf_name, metrics in results.items():
            if metrics is not None:
                combined_data.append({
                    'Dataset': dataset_name,
                    'Classifier': clf_name,
                    'Accuracy': metrics['Acc'],
                    'Sensitivity': metrics['Sen'],
                    'Specificity': metrics['Spe'],
                    'Precision': metrics['Pre'],
                    'MCC': metrics['Mcc'],
                    'AUROC': metrics['AUROC'],
                    'AUPRC': metrics['AUPRC'],
                    'F1_Score': metrics['F1']
                })

    if combined_data:
        df = pd.DataFrame(combined_data)
        csv_path = f"{output_dir}/classifier_performance.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCombined performance results saved to: {csv_path}")


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
    os.environ['OMP_NUM_THREADS'] = '2'  # 限制OpenMP线程数
    os.environ['OPENBLAS_NUM_THREADS'] = '2'  # 限制OpenBLAS线程数

    # 设置Windows兼容性
    if os.name == 'nt':
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    main()
