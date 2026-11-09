from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def init_classifiers(seed=42):
    classifiers = {
        'Random Forest': RandomForestClassifier(
            random_state=seed, n_jobs=2, n_estimators=100,
            max_depth=15, max_samples=0.5, verbose=0),
        'Logistic Regression': LogisticRegression(
            random_state=seed, max_iter=500, n_jobs=1, solver='lbfgs', verbose=0),
        'Decision Tree': DecisionTreeClassifier(random_state=seed, max_depth=15),
        'Gaussian NB': GaussianNB(),
        'MLP': MLPClassifier(
            random_state=seed, max_iter=300, hidden_layer_sizes=(128, 64),
            early_stopping=True, solver='adam', verbose=0),
    }
    try:
        from xgboost import XGBClassifier
        classifiers['XGBoost'] = XGBClassifier(
            random_state=seed, n_jobs=2, n_estimators=100, max_depth=8,
            tree_method='hist', verbosity=0)
    except ImportError:
        pass
    return classifiers


def train(clf, X, y, sample_weight=None):
    if sample_weight is not None:
        try:
            clf.fit(X, y, sample_weight=sample_weight)
            return clf
        except TypeError:
            pass
    clf.fit(X, y)
    return clf


def predict_proba(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X).astype(float)
