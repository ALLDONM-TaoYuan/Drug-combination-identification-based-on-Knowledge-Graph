import numpy as np
import pandas as pd
import h5py
import json
import os
import xgboost as xgb
import warnings
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

class EnhancedH5Loader:
    def __init__(self, h5_path, metadata_path, set_name='train'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.positive_batches = metadata[set_name]['positive']
        self.negative_batches = metadata[set_name]['negative']
        self.h5_path = h5_path
        self.set_name = set_name

        random.shuffle(self.positive_batches)
        random.shuffle(self.negative_batches)

    def load_samples(self, max_samples=1000000, random_seed=42):
        features = []
        labels = []

        if random_seed is not None:
            np.random.seed(random_seed)

        with h5py.File(self.h5_path, 'r') as f:
            for batch in tqdm(self.positive_batches, desc=f"Loading {self.set_name} positive samples"):
                group = f[batch['group_path']]
                batch_features = group['features'][:]
                batch_labels = group['labels'][:]

                n_samples = min(len(batch_features), max_samples // len(self.positive_batches))
                if n_samples > 0:
                    indices = np.random.choice(len(batch_features), n_samples, replace=False)
                    features.append(batch_features[indices])
                    labels.append(batch_labels[indices])

            for batch in tqdm(self.negative_batches, desc=f"Loading {self.set_name} negative samples"):
                group = f[batch['group_path']]
                batch_features = group['features'][:]
                batch_labels = group['labels'][:]

                n_samples = min(len(batch_features), max_samples // len(self.negative_batches))
                if n_samples > 0:
                    indices = np.random.choice(len(batch_features), n_samples, replace=False)
                    features.append(batch_features[indices])
                    labels.append(batch_labels[indices])

        if features:
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)

        indices = np.random.permutation(len(features))
        return features[indices][:max_samples], labels[indices][:max_samples]

def setup_xgboost_params():
    params = {
        'tree_method': 'hist',
        'device': 'cpu',
        'nthread': 20,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc', 'error'],
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 20,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
        'gamma': 1.0,
        'seed': 42,
        'verbosity': 0,
    }
    return params

def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=2000):
    params = setup_xgboost_params()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        evals_result=evals_result,
        early_stopping_rounds=500,
        verbose_eval=100
    )

    return model, evals_result

def calculate_metrics(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc_score = roc_auc_score(y_true, y_pred_proba)
    aupr_score = average_precision_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'Accuracy': acc,
        'Sensitivity': sen,
        'Specificity': spe,
        'Precision': pre,
        'F1': f1,
        'AUC': auc_score,
        'AUPR': aupr_score,
        'MCC': mcc
    }

def plot_performance_summary(evals_result, train_metrics, val_metrics, test_metrics,
                             y_train_pred_proba, y_val_pred_proba, y_test_pred_proba,
                             y_train, y_val, y_test, output_dir='model_results'):
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 8))

    color_train = '#1f77b4'
    color_val = '#ff7f0e'
    color_test = '#2ca02c'

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          wspace=0.2, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    if 'train' in evals_result and 'error' in evals_result['train']:
        train_error = evals_result['train']['error']
        train_accuracy = 1 - np.array(train_error)

        if 'val' in evals_result and 'error' in evals_result['val']:
            val_error = evals_result['val']['error']
            val_accuracy = 1 - np.array(val_error)

            iterations = range(1, len(train_accuracy) + 1)
            ax1.plot(iterations, train_accuracy, color=color_train, linewidth=2)
            ax1.plot(iterations, val_accuracy, color=color_val, linewidth=2)

            ax1.set_xlabel('Iteration', fontsize=24, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=24, fontweight='bold')
            ax1.grid(False)
            ax1.set_xlim([-0.05, len(train_accuracy)])
            ax1.set_ylim([0.5, 1.05])
            ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax1.tick_params(axis='both', which='major', labelsize=20)
            ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes,
                     fontsize=18, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(gs[1, 0])
    if 'train' in evals_result and 'logloss' in evals_result['train']:
        train_loss = evals_result['train']['logloss']
        ax2.plot(range(1, len(train_loss) + 1), train_loss, color=color_train, linewidth=2)

        if 'val' in evals_result and 'logloss' in evals_result['val']:
            val_loss = evals_result['val']['logloss']
            ax2.plot(range(1, len(val_loss) + 1), val_loss, color=color_val, linewidth=2)

        ax2.set_xlabel('Iteration', fontsize=24, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=24, fontweight='bold')
        ax2.grid(False)
        ax2.set_xlim([-0.05, len(train_loss)])
        ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes,
                 fontsize=18, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(gs[:, 1])

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)

    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_pred_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)

    ax3.plot(fpr_train, tpr_train, color=color_train, linewidth=2)
    ax3.plot(fpr_val, tpr_val, color=color_val, linewidth=2)
    ax3.plot(fpr_test, tpr_test, color=color_test, linewidth=2)

    ax3.plot(recall_train, precision_train, color=color_train, linewidth=2)
    ax3.plot(recall_val, precision_val, color=color_val, linewidth=2)
    ax3.plot(recall_test, precision_test, color=color_test, linewidth=2)

    ax3.set_xlabel('False Positive Rate / Recall', fontsize=24, fontweight='bold')
    ax3.set_ylabel('True Positive Rate / Precision', fontsize=24, fontweight='bold')
    ax3.grid(False)
    ax3.set_xlim([-0.025, 1.025])
    ax3.set_ylim([0, 1.025])
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes,
             fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    legend_elements = [
        Patch(facecolor=color_train, edgecolor='black', label='Training'),
        Patch(facecolor=color_val, edgecolor='black', label='Validation'),
        Patch(facecolor=color_test, edgecolor='black', label='Test')
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=3,
               fontsize=24,
               frameon=False)

    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def main():
    h5_path = './feature_batches/dataset.h5'
    metadata_path = './feature_batches/dataset_metadata.json'

    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found: {h5_path}")
        return

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return

    print("Loading data with enhanced randomization...")

    train_loader = EnhancedH5Loader(h5_path, metadata_path, 'train')
    val_loader = EnhancedH5Loader(h5_path, metadata_path, 'val')
    test_loader = EnhancedH5Loader(h5_path, metadata_path, 'test')

    max_train_samples = 1000000
    max_val_samples = 200000
    max_test_samples = 200000

    print(f"Loading training samples...")
    X_train, y_train = train_loader.load_samples(max_samples=max_train_samples, random_seed=42)

    print(f"Loading validation samples...")
    X_val, y_val = val_loader.load_samples(max_samples=max_val_samples, random_seed=43)

    print(f"Loading test samples...")
    X_test, y_test = test_loader.load_samples(max_samples=max_test_samples, random_seed=44)

    print(f"Training samples: {len(X_train):,} (Positive: {np.sum(y_train == 1):,}, Negative: {np.sum(y_train == 0):,})")
    print(f"Validation samples: {len(X_val):,} (Positive: {np.sum(y_val == 1):,}, Negative: {np.sum(y_val == 0):,})")
    print(f"Test samples: {len(X_test):,} (Positive: {np.sum(y_test == 1):,}, Negative: {np.sum(y_test == 0):,})")

    print("\nTraining XGBoost model with 20000 iterations...")
    model, evals_result = train_xgboost(X_train, y_train, X_val, y_val, n_estimators=500)

    print("\nEvaluating on training set...")
    dtrain = xgb.DMatrix(X_train)
    y_train_pred_proba = model.predict(dtrain)
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

    print("\nEvaluating on validation set...")
    dval = xgb.DMatrix(X_val)
    y_val_pred_proba = model.predict(dval)
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)

    print("\nEvaluating on test set...")
    dtest = xgb.DMatrix(X_test)
    y_test_pred_proba = model.predict(dtest)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

    output_dir = 'model_results'
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving model and results...")
    model.save_model(f'{output_dir}/xgboost_model.json')

    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    metrics_df.insert(0, 'Dataset', ['Training', 'Validation', 'Test'])
    metrics_df.to_csv(f'{output_dir}/model_metrics.csv', index=False)

    config = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'feature_dim': X_train.shape[1],
        'n_estimators': 2000,
        'early_stopping_rounds': 500,
        'random_seed': 42
    }

    with open(f'{output_dir}/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nPlotting performance summary with optimized layout...")
    fig1 = plot_performance_summary(
        evals_result, train_metrics, val_metrics, test_metrics,
        y_train_pred_proba, y_val_pred_proba, y_test_pred_proba,
        y_train, y_val, y_test, output_dir
    )

    print("\n" + "=" * 60)
    print("Model Training Results")
    print("=" * 60)

    for dataset_name, metrics in zip(['Training', 'Validation', 'Test'], [train_metrics, val_metrics, test_metrics]):
        print(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()