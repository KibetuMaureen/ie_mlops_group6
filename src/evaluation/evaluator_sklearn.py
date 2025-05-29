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
    Uses sklearn's built-in metric functions.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = None

        logging.info("[SKLEARN] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %s",
                     acc, precision, recall, f1, f"{auc:.4f}" if auc else "N/A")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
            "confusion_matrix": cm
        }

    except Exception as e:
        logging.error(f"Sklearn evaluation failed: {e}")
        raise
