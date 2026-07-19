import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True,
                fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(
        'correlation_heatmap.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()


def plot_regression_diagnostics(y_test, y_pred, residuals):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', linewidth=2
    )
    axes[0].set_title('Actual vs Predicted')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')

    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')

    axes[2].hist(residuals, bins=30, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_title('Residual Distribution')
    axes[2].set_xlabel('Residual')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(
        'regression_diagnostics.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()


def plot_coefficients(feature_names, coefficients):
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, coefficients)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.savefig(
        'feature_coefficients.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
