"""
Evaluate classification model performance using scikit-learn metrics.

Provides a function to calculate accuracy, precision, recall, F1-score,
ROC AUC, and confusion matrix from true and predicted labels.
"""


import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

logging.basicConfig(level=logging.INFO)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model predictions using common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1-score,
              ROC AUC (if applicable), and the confusion matrix.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average='binary', zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average='binary', zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average='binary', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:  # pylint: disable=broad-except
            auc = None

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        logging.info(
            "[SKLEARN] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | "
            "F1: %.4f | AUC: %s",
            acc, precision, recall, f1, auc_str
        )

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
            "confusion_matrix": cm
        }

    except Exception as e:
        logging.error("Sklearn evaluation failed: %s", e)
        raise
