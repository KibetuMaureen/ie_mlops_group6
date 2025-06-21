import argparse
import joblib
from evaluator_sklearn import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_true_path", type=str, required=True)
    parser.add_argument("--y_pred_path", type=str, required=True)
    args = parser.parse_args()

    # Load inputs
    y_true = joblib.load(args.y_true_path)
    y_pred = joblib.load(args.y_pred_path)

    # Run evaluation
    results = evaluate_model(y_true, y_pred)

    # Output results
    for key, value in results.items():
        print(f"{key}: {value}")
