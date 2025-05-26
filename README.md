# ie_mlops_group6

## Credit Card Fraud Detection MLOps Pipeline

This project implements a modular, end-to-end MLOps pipeline for credit card fraud detection using Python, scikit-learn, XGBoost, and best practices for reproducibility and leakage-proof modeling.

---

## 📁 Project Structure

```
ie_mlops_group6/
│
├── config.yaml                # Main configuration file
├── environment.yml            # Conda environment with dependencies
├── data/
│   └── raw/                   # Raw data files (e.g., fraudTrain.csv)
├── models/                    # Saved models and metrics
├── logs/                      # Log files and validation reports
├── src/
│   ├── main.py                # Pipeline entry point
│   ├── data_loader/           # Data loading utilities
│   ├── data_validation/       # Data validation utilities
│   ├── preprocessing/         # Preprocessing pipeline (feature engineering, encoding, etc.)
│   └── model/                 # Model training, optimization, and evaluation
└── tests/                     # Unit tests
```

---

## 🚀 How to Run

1. **Clone the repository and navigate to the project root:**

    ```bash
    git clone <repo-url>
    cd ie_mlops_group6
    ```

2. **Set up the environment:**

    ```bash
    conda env create -f environment.yml
    conda activate fraud_detection_env
    ```

3. **Place your raw data file** (e.g., `fraudTrain.csv`) in `data/raw/`.

4. **Edit `config.yaml`** as needed to match your data and experiment settings.

5. **Run the pipeline:**

    ```bash
    python -m src.main --config config.yaml --stage all
    ```

    - Use `--stage data` to only load and validate data.
    - Use `--stage train` to run preprocessing and model training.

---

## ⚙️ Pipeline Stages

- **Data Loading:** Reads raw data from CSV or other sources.
- **Data Validation:** Checks schema, types, and required columns.
- **Preprocessing:** Feature engineering, encoding, scaling, and leakage-proof transformations using sklearn pipelines.
- **Model Training:** Hyperparameter optimization (Bayesian), training, evaluation, and artifact saving (XGBoost or other models).
- **Logging:** All steps are logged for traceability.

---

## 🧪 Testing

Unit tests are provided in the `tests/` directory.  
Run all tests with:

```bash
pytest tests/
```

---

## 📄 Configuration

All pipeline settings (features, model parameters, paths, etc.) are controlled via `config.yaml`.  
Edit this file to customize your experiment.

---

## 📝 Authors

- Group 6, IE MBD ML Ops

---

## 📬 Contact

For questions or contributions, please open an issue or contact the project maintainers.
