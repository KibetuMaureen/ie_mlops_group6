import sys
import os
import numpy as np
import pytest

# Add src/ to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import your module
from evaluation import evaluator_sklearn

def test_typical_case():
    """Test with typical binary classification case."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    results = evaluator_sklearn.evaluate_model(y_true, y_pred)

    assert isinstance(results, dict)
    assert round(results["accuracy"], 2) == 0.67
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert results["confusion_matrix"].tolist() == [[2, 1], [1, 2]]

def test_all_predictions_wrong():
    """Test when all predictions are incorrect."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    results = evaluator_sklearn.evaluate_model(y_true, y_pred)

    assert results["accuracy"] == 0.0
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["f1_score"] == 0.0

def test_invalid_labels_raise_error():
    """Test that invalid prediction values raise an error."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.3, 0.9])  # invalid float labels

    with pytest.raises(ValueError):
        evaluator_sklearn.evaluate_model(y_true, y_pred)
