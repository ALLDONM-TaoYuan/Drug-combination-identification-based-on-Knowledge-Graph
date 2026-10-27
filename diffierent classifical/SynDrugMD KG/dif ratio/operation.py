import numpy as np
import pandas as pd
import h5py
import os
import xgboost as xgb
import warnings
import gc
import random
import time
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


class H5BalancedDataLoader:
    """Loader for balanced HDF5 datasets"""

    def __init__(self, h5_path):
        self.h5_path = h5_path

    def load_all_data(self):
        """Load all data (train+val+test) for cross-validation"""
        with h5py.File(self.h5_path, 'r') as f:
            # Load train data
            X_train = f['train']['features'][:]
            y_train = f['train']['labels'][:]

            # Load validation data
            X_val = f['val']['features'][:]
            y_val = f['val']['labels'][:]

            # Load test data
            X_test = f['test']['features'][:]
            y_test = f['test']['labels'][:]

            # Combine all data
            X_all = np.concatenate([X_train, X_val, X_test], axis=0)
            y_all = np.concatenate([y_train, y_val, y_test], axis=0)

            return X_all, y_all


def setup_xgboost_params():
    """Setup XGBoost parameters for CPU only"""
    return {
        'tree_method': 'hist',
        'device': 'cpu',
        'nthread': 256,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc', 'error'],
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'gamma': 0.2,
        'seed': 42,
        'verbosity': 0,
        'max_bin': 256,
    }


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=500, early_stopping_rounds=100):
    """Train XGBoost model using CPU only"""
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
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    return model, evals_result


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive performance metrics"""
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spe = 0

    auc_score = roc_auc_score(y_true, y_pred_proba)
    aupr_score = average_precision_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'Accuracy': acc,
        'Sensitivity': sen,
        'Specificity': spe,
        'Precision': pre,
        'F1_Score': f1,
        'AUC': auc_score,
        'AUPR': aupr_score,
        'MCC': mcc
    }


def evaluate_model(model, X, y):
    """Evaluate model on given data"""
    dmatrix = xgb.DMatrix(X)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return calculate_metrics(y, y_pred, y_pred_proba)


def run_cross_validation(X, y, n_folds=5, n_estimators=500, early_stopping_rounds=100):
    """Run n-fold cross-validation and return test metrics"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    fold_times = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold_idx}/{n_folds}...")

        # Split data
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Further split train_val into train and validation
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf_inner.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # Train model
        start_time = time.time()
        model, _ = train_xgboost(X_train, y_train, X_val, y_val, n_estimators, early_stopping_rounds)
        fold_time = time.time() - start_time

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Fold'] = fold_idx
        metrics['Train_Time'] = fold_time
        fold_metrics.append(metrics)
        fold_times.append(fold_time)

        print(f"    Test AUC: {metrics['AUC']:.4f}, Time: {fold_time:.1f}s")

        # Clean memory
        gc.collect()

    # Calculate average metrics
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if key != 'Fold':
            values = [m[key] for m in fold_metrics]
            avg_metrics[f'Avg_{key}'] = np.mean(values)
            avg_metrics[f'Std_{key}'] = np.std(values)

    avg_metrics['Avg_Train_Time'] = np.mean(fold_times)
    avg_metrics['Total_Train_Time'] = np.sum(fold_times)

    return fold_metrics, avg_metrics


def run_all_ratios_cv():
    """Run 5-fold cross-validation for all ratios"""

    data_dir = './balanced_datasets'
    output_dir = './cv_results'
    ratios = [1, 2, 3, 4, 5]
    n_folds = 5

    print("=" * 60)
    print(f"5-Fold Cross-Validation Performance Testing")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing ratios: {[f'1:{r}' for r in ratios]}")
    print(f"N_folds: {n_folds}, N_estimators: 500, Early_stopping: 100")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for ratio in ratios:
        print(f"\nProcessing ratio 1:{ratio}...")

        h5_file = os.path.join(data_dir, f'ratio_1_{ratio}.h5')

        if not os.path.exists(h5_file):
            print(f"Error: Data file not found: {h5_file}")
            continue

        try:
            # Load all data
            loader = H5BalancedDataLoader(h5_file)
            X_all, y_all = loader.load_all_data()

            n_pos = np.sum(y_all == 1)
            n_neg = np.sum(y_all == 0)
            print(f"Dataset loaded: {len(X_all):,} samples")
            print(f"Positive/Negative: {n_pos:,}/{n_neg:,}")
            print(f"Feature dimension: {X_all.shape[1]}")

            # Run cross-validation
            start_time = time.time()
            fold_metrics, avg_metrics = run_cross_validation(X_all, y_all, n_folds)
            total_time = time.time() - start_time

            # Save fold results
            fold_df = pd.DataFrame(fold_metrics)
            fold_csv = os.path.join(output_dir, f'ratio_1_{ratio}_fold_results.csv')
            fold_df.to_csv(fold_csv, index=False)

            # Add ratio info to average metrics
            avg_metrics['Ratio'] = f'1:{ratio}'
            avg_metrics['Total_Samples'] = len(X_all)
            avg_metrics['Positive_Samples'] = n_pos
            avg_metrics['Negative_Samples'] = n_neg
            avg_metrics['Total_Time'] = total_time
            all_results.append(avg_metrics)

            # Print average results
            print(f"\nAverage results for ratio 1:{ratio}:")
            print(f"  Test AUC:  {avg_metrics['Avg_AUC']:.4f} ± {avg_metrics['Std_AUC']:.4f}")
            print(f"  Test Accuracy:  {avg_metrics['Avg_Accuracy']:.4f} ± {avg_metrics['Std_Accuracy']:.4f}")
            print(f"  Test F1:  {avg_metrics['Avg_F1_Score']:.4f} ± {avg_metrics['Std_F1_Score']:.4f}")
            print(f"  Total training time: {total_time:.1f}s")
            print("-" * 40)

        except Exception as e:
            print(f"Error in ratio 1:{ratio}: {e}")
        finally:
            gc.collect()

    # Generate summary
    if all_results:
        print("\n" + "=" * 60)
        print("Summary of 5-Fold Cross-Validation Results")
        print("=" * 60)

        summary_df = pd.DataFrame(all_results)

        # Reorder columns
        cols = ['Ratio', 'Total_Samples', 'Positive_Samples', 'Negative_Samples',
                'Total_Time', 'Total_Train_Time', 'Avg_Train_Time']
        metric_cols = [c for c in summary_df.columns if c not in cols]
        cols_order = cols + sorted(metric_cols)
        summary_df = summary_df[cols_order]

        # Save summary
        summary_csv = os.path.join(output_dir, 'cv_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary saved to: {summary_csv}")

        # Print summary table
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        print("\nPerformance Summary (mean ± std):")

        # Display key metrics
        display_cols = ['Ratio', 'Avg_Accuracy', 'Avg_AUC', 'Avg_AUPR',
                        'Avg_F1_Score', 'Avg_MCC', 'Total_Time']
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()

        # Format for display
        for col in display_df.columns:
            if col != 'Ratio' and 'Time' not in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            elif 'Time' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")

        print(display_df.to_string(index=False))

        # Find best ratio by AUC
        if 'Avg_AUC' in summary_df.columns:
            best_idx = summary_df['Avg_AUC'].idxmax()
            best_ratio = summary_df.loc[best_idx, 'Ratio']
            best_auc = summary_df.loc[best_idx, 'Avg_AUC']
            best_std = summary_df.loc[best_idx, 'Std_AUC']
            print(f"\nBest ratio (by AUC): {best_ratio} (AUC = {best_auc:.4f} ± {best_std:.4f})")

    print("\n" + "=" * 60)
    print("All cross-validation experiments completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_ratios_cv()