"""
visualisations.py
"""



# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Models evaluation
from sklearn.metrics import (
    ConfusionMatrixDisplay,
)



def plot_na_heatmap(df, mean_=False):
    if mean_:
        plt.figure(figsize=(16, 4))
        sns.heatmap(
            100*df.isna().mean().to_frame().T,
            annot=True, cbar=False, fmt=".1f"
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(16, 7))
        sns.heatmap(df.isna(), cbar=False)
        plt.yticks([])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def plot_numericals_kde(df, numerical_cols):
    n_cols = 5
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    axes = axes.ravel()
    for i, col in enumerate(numerical_cols):
        sns.kdeplot(
            data=df[col].dropna(), ax=axes[i], fill=True
        )
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # hide unused axes
    for ax in axes[len(numerical_cols):]:
        ax.set_visible(False)
    plt.suptitle('KDE of numerical features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_evaluate_model(metrics, model_name, threshold):
    cr, cm = metrics['classification_report'], metrics['cm']
    fpr, tpr = metrics['fpr'], metrics['tpr']
    prob_true, prob_pred = metrics['prob_true'], metrics['prob_pred']
    importances = metrics['importances']

    train_sizes = metrics['train_sizes']
    train_mean = metrics['train_mean']
    valid_mean = metrics['valid_mean']

    print("\nClassification report:\n")
    print(cr)
    print()

    # ROC curve
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(model_name, fontsize=14, fontweight='bold')
    axes[0, 0].plot(fpr, tpr, label=f"AUC = {metrics['auc_test']:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("FPR")
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Calibration curve
    axes[0, 1].plot(prob_pred, prob_true, marker='o', label='Model')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[0, 1].set_title("Calibration Curve")
    axes[0, 1].set_xlabel("Predicted PD")
    axes[0, 1].set_ylabel("Observed PD")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # Learning curve
    axes[0, 2].plot(train_sizes, train_mean, 'o-', label="Train AUC")
    axes[0, 2].plot(train_sizes, valid_mean, 'o-', label="Valid AUC")
    axes[0, 2].set_title("Learning Curve")
    axes[0, 2].set_xlabel("Training Size")
    axes[0, 2].set_ylabel("ROC AUC")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[1, 2].axis('off')
    # Confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=['Non-Default', 'Default']).plot(
        ax=axes[1, 0],
        colorbar=False
    )
    axes[1, 0].set_title(f"Confusion Matrix (threshold={threshold})")
    # Importance
    importances.tail(15).plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_title("Top 15 Positive Drivers")
    axes[1, 1].set_xlabel("Coefficient")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_step):
    # Build comparison dataframe
    comparison_df = pd.DataFrame([
        {
            'model': name,
            'auc_train': metrics['auc_train'],
            'auc_test': metrics['auc_test'],
            'cv_auc': metrics['cv_auc'],
            'cv_auc_std': metrics['cv_auc_std'],
            'avg_precision': metrics['avg_precision'],
            'brier': metrics['brier'],
            'fit_time_sec': metrics['fit_time_sec'],
        }
        for name, metrics in results_step.items()
    ])

    # Generalization gap
    comparison_df['overfit_gap'] = (comparison_df['auc_train'] - comparison_df['auc_test'])
    models = comparison_df['model']
    # AUC Train vs Test
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Model Comparison Dashboard", fontsize=16, fontweight='bold')
    axes[0, 0].plot(models, comparison_df['auc_train'], marker='o', label='Train AUC')
    axes[0, 0].plot(models, comparison_df['auc_test'], marker='o', label='Test AUC')
    axes[0, 0].set_title("AUC (Train vs Test)")
    axes[0, 0].set_ylabel("AUC")
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # CV AUC with uncertainty
    axes[0, 1].errorbar(
        models, comparison_df['cv_auc'], yerr=comparison_df['cv_auc_std'], fmt='o',
        capsize=6, markersize=8, elinewidth=1.5, ecolor='red', label='CV AUC')
    axes[0, 1].set_title("Cross-Validation AUC (mean +- std)")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].grid(True)
    # Precision vs Brier Score
    axes[1, 0].plot(models, comparison_df['avg_precision'], marker='o', label='Avg Precision')
    axes[1, 0].plot(models, comparison_df['brier'], marker='o', label='Brier Score')
    axes[1, 0].set_title("Precision vs Calibration Quality")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].tick_params(axis='x', rotation=30)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # Overfitting + Training Time
    ax2 = axes[1, 1]
    ax2.bar(models, comparison_df['overfit_gap'], alpha=0.6, label='Overfit Gap')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(models, comparison_df['fit_time_sec'], marker='o', color='red', label='Fit Time (sec)')
    ax2.set_title("Overfitting & Training Cost")
    ax2.set_ylabel("AUC Gap")
    ax2_twin.set_ylabel("Time (sec)")
    ax2.tick_params(axis='x', rotation=30)
    # combined legend
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()



def plot_framework_model_comparison(results_dict):
    def build_df(results, label):
        df = pd.DataFrame([
            {'model': name, 'framework': label, 'auc_train': m['auc_train'],
             'auc_test': m['auc_test'], 'cv_auc': m['cv_auc'],
             'cv_auc_std': m['cv_auc_std'], 'avg_precision': m['avg_precision'],
             'brier': m['brier'], 'fit_time_sec': m['fit_time_sec'],
            } for name, m in results.items()
        ])
        df['overfit_gap'] = df['auc_train'] - df['auc_test']
        return df
    # Build full dataframe
    dfs = [build_df(res, name) for name, res in results_dict.items()]
    df = pd.concat(dfs, ignore_index=True)

    frameworks = list(results_dict.keys())
    models = df['model'].unique()
    x = np.arange(len(models))
    width = 0.8 / len(frameworks)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Framework Comparison Dashboard", fontsize=16, fontweight='bold')
    colors = plt.cm.tab10.colors
    # AUC TRAIN
    axes[0, 0].set_title("AUC Train")
    axes[0, 0].grid(True)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[0, 0].plot(models, sub['auc_train'], marker='o', label=fw, color=colors[i])
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].legend()
    # AUC TEST
    axes[0, 1].set_title("AUC Test")
    axes[0, 1].grid(True)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[0, 1].plot(models, sub['auc_test'], marker='o', label=fw, color=colors[i])
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].legend()
    # CV AUC ± STD
    axes[0, 2].set_title("CV AUC (mean ± std)")
    axes[0, 2].grid(True)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[0, 2].errorbar(models, sub['cv_auc'], yerr=sub['cv_auc_std'],
            fmt='o', capsize=4, label=fw, color=colors[i])
    axes[0, 2].tick_params(axis='x', rotation=30)
    axes[0, 2].legend()
    # Precision + Brier
    axes[1, 0].set_title("Precision vs Calibration (Brier)")
    axes[1, 0].grid(True)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[1, 0].plot(models, sub['avg_precision'], marker='o',
                        label=f"{fw} Precision", color=colors[i])
        axes[1, 0].plot(models, sub['brier'], marker='x',
                        linestyle='--', label=f"{fw} Brier", color=colors[i])
    axes[1, 0].tick_params(axis='x', rotation=30)
    axes[1, 0].legend()
    # Overfitting Gap
    axes[1, 1].set_title("Overfitting Gap")
    axes[1, 1].grid(True)
    width = 0.8 / len(frameworks)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[1, 1].bar(x + i * width, sub['overfit_gap'],
            width, label=fw, color=colors[i])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=30)
    axes[1, 1].legend()
    # Training Time
    axes[1, 2].set_title("Training Time (seconds)")
    axes[1, 2].grid(True)
    for i, fw in enumerate(frameworks):
        sub = df[df['framework'] == fw].set_index("model").reindex(models)
        axes[1, 2].bar(x + i * width, sub['fit_time_sec'],
            width, label=fw, color=colors[i])
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models, rotation=30)
    axes[1, 2].legend()
    plt.tight_layout()
    plt.show()