"""
models.py
"""



import time

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import pandas as pd

# Model selection
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve
)

# Models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.cluster.hierarchy import linkage, dendrogram

# Models evaluation
from sklearn.metrics import (
    precision_recall_curve, brier_score_loss,
    roc_auc_score, roc_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score, f1_score
)
from sklearn.calibration import calibration_curve



# Config
SEED = 42
np.random.seed(SEED)



def time_train_test_split(framework_df, date_col="date", target_col="default", test_size=0.2):
    df = framework_df.sort_values(date_col).copy()
    split_index = int(len(df) * (1 - test_size))
    split_date = df.iloc[split_index][date_col]

    train_df = df[df[date_col] <= split_date]
    test_df = df[df[date_col] > split_date]

    X_train = train_df.drop(columns=[target_col, date_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, date_col])
    y_test = test_df[target_col]

    train_dates = train_df[date_col]
    test_dates = test_df[date_col]
    print(f"Train : {X_train.shape} | Test: {X_test.shape}")
    print(f"Split date : {split_date}")
    return X_train, X_test, y_train, y_test, train_dates, test_dates, split_date



def evaluate_model(
    model, X_train, y_train, X_test, y_test, model_name, threshold=0.5
):
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]
    y_pred_test  = (y_prob_test >= threshold).astype(int)

    # Learning curve
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=SEED
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, scoring='roc_auc', n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    )
    # Importance
    if hasattr(model, "coef_"):
        importances = pd.Series(model.coef_[0], index=X_train.columns)
    elif hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(key=np.abs)
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    # calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    # classification report
    cr = classification_report(y_test, y_pred_test)
    metrics = {
        'model': model, 'model_name': model_name,
        'auc_train': roc_auc_score(y_train, y_prob_train),
        'auc_test': roc_auc_score(y_test, y_prob_test),
        'cv_auc': cv_scores.mean(), 'cv_auc_std': cv_scores.std(),
        'avg_precision_train': average_precision_score(y_train, y_prob_train),
        'avg_precision': average_precision_score(y_test, y_prob_test),
        'brier': brier_score_loss(y_test, y_prob_test),
        'cv_scores': cv_scores, 'importances': importances, 'fpr': fpr, 'tpr': tpr,
        'prob_true': prob_true, 'prob_pred': prob_pred, 'cm': cm, 'classification_report': cr,
        'train_sizes': train_sizes, 'train_mean': train_mean, 'valid_mean': valid_mean
    }
    return metrics


def test_candidate_models(X_train, y_train, X_test, y_test, threshold):
    cw = "balanced"
    models_dict = {
        # LINEAR MODELS
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight=cw, n_jobs=-1,
            random_state=SEED
        ),
        "lda": LinearDiscriminantAnalysis(),
        # TREES MODELS
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, class_weight=cw, random_state=SEED
        ),
        # BOOSTING MODELS
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, n_jobs=-1, random_state=SEED
        )
    }

    results_step = {}
    for name, model in models_dict.items():
        try:
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name, threshold)

            end_time = time.perf_counter()
            exec_time = end_time - start_time
            metrics['fit_time_sec'] = exec_time
            results_step[name] = metrics
            print(f"Evaluated model '{name}' in {exec_time:.3f}seconds")
        except Exception as e:
            print(f"Error while evaluating '{name}': {e}")
            continue
    return results_step


def evaluate_thresholds(X_train, y_train, X_test, y_test, plot_=False):  
    model_test = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="saga", n_jobs=-1
    )
    model_test.fit(X_train, y_train)
    # predicted probabilities
    y_prob = model_test.predict_proba(X_test)[:, 1]
    results = []
    thresholds = np.arange(0.1, 0.91, 0.05)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(y_test, y_pred)
        results.append({
            "threshold": threshold, "accuracy": accuracy,
            "precision": precision, "recall": recall,
            "f1_score": f1, "defaults_predicted": y_pred.sum()
        })
    results = pd.DataFrame(results)
    
    if plot_:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in metrics:
            axes[0].plot(results["threshold"], results[metric],
                marker="o", label=metric)
        axes[0].set_title("Threshold Impact on Classification Metrics")
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        # Number of predicted defaults
        axes[1].plot(
            results["threshold"], results["defaults_predicted"],
            marker="o", linewidth=2)
        axes[1].set_title("Predicted Defaults vs Threshold")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Number of predicted defaults")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return results


def compare_framework_models_eval(results_dict, threshold=0.5, model_name=None):
    frameworks = list(results_dict.keys())
    # find common models across frameworks
    common_models = set(results_dict[frameworks[0]].keys())
    for fw in frameworks[1:]:
        common_models &= set(results_dict[fw].keys())
    common_models = sorted(list(common_models))

    # default: take first model if not specified
    if model_name is None:
        model_name = common_models[0]
    m = {fw: results_dict[fw][model_name] for fw in frameworks}

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"{model_name}: Framework Comparison",
        fontsize=15, fontweight='bold')
    # ROC CURVE
    for i, fw in enumerate(frameworks):
        axes[0, 0].plot(m[fw]['fpr'], m[fw]['tpr'],
            label=f"{fw} (AUC={m[fw]['auc_test']:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("FPR")
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # CALIBRATION CURVE
    for i, fw in enumerate(frameworks):
        axes[0, 1].plot(m[fw]['prob_pred'],  m[fw]['prob_true'],
            marker='o', label=fw)
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_title("Calibration Curve")
    axes[0, 1].set_xlabel("Predicted PD")
    axes[0, 1].set_ylabel("Observed PD")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # LEARNING CURVE
    for i, fw in enumerate(frameworks):
        axes[0, 2].plot(m[fw]['train_sizes'], m[fw]['train_mean'],
                        'o-', label=f"{fw} Train")
        axes[0, 2].plot(m[fw]['train_sizes'], m[fw]['valid_mean'],
                        'o--', label=f"{fw} Valid")
    axes[0, 2].set_title("Learning Curve")
    axes[0, 2].set_xlabel("Training Size")
    axes[0, 2].set_ylabel("ROC AUC")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True)
    # CONFUSION MATRICES
    for i, fw in enumerate(frameworks):
        sns.heatmap(m[fw]['cm'], annot=True, fmt='d', cbar=False, ax=axes[1, i],
            cmap='Blues' if i == 0 else 'Greens',
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'])
        axes[1, i].set_title(f"{fw}")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("True")
    # FEATURE IMPORTANCE COMPARISON
    imp_df = pd.concat({fw: m[fw]['importances'].tail(10) for fw in frameworks},
        axis=1).fillna(0)
    imp_df.plot(kind='barh', ax=axes[1, 2])
    axes[1, 2].axvline(0, color='black', linestyle='--')
    axes[1, 2].set_title("Top Feature Importances")
    axes[1, 2].set_xlabel("Importance")
    plt.tight_layout()
    plt.show()

