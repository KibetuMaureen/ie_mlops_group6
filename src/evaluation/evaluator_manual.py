import logging
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO)

def evaluate_model(y_true, y_pred):
    """
    Manually computes evaluation metrics using confusion matrix.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = None

        logging.info("[MANUAL] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %s",
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
        logging.error(f"Manual evaluation failed: {e}")
        raise

