"""
Intent Classifier Training Pipeline with MLflow Tracking
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Optional MLflow tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def train_model(
    data_path: Path,
    artifacts_dir: Path,
    experiment_name: str = "Intent-Classifier-Manjunath",
    author: str = "Manjunath",
    random_state: int = 42,
) -> dict:
    print(f"==================================================")
    print(f" Training Intent Classifier Pipeline")
    print(f" Author: {author}")
    print(f"==================================================")

    # 1. Load data
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    X = df["text"].astype(str)
    y = df["intent"].astype(str)
    num_samples = len(df)
    unique_intents = sorted(y.unique().tolist())
    print(f"Loaded {num_samples} samples across {len(unique_intents)} intents: {unique_intents}")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Define Pipeline (TF-IDF + Logistic Regression)
    params = {
        "vect__ngram_range": (1, 2),
        "vect__sublinear_tf": True,
        "vect__min_df": 1,
        "clf__C": 2.0,
        "clf__solver": "lbfgs",
        "clf__max_iter": 1000,
        "clf__random_state": random_state,
    }

    pipeline = Pipeline([
        ("vect", TfidfVectorizer(
            ngram_range=params["vect__ngram_range"],
            sublinear_tf=params["vect__sublinear_tf"],
            min_df=params["vect__min_df"],
        )),
        ("clf", LogisticRegression(
            C=params["clf__C"],
            solver=params["clf__solver"],
            max_iter=params["clf__max_iter"],
            random_state=params["clf__random_state"],
        )),
    ])

    # 4. Train Model
    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_acc = float(accuracy_score(y_train, y_pred_train))
    test_acc = float(accuracy_score(y_test, y_pred_test))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted", zero_division=0)

    clf_report = classification_report(y_test, y_pred_test, zero_division=0)
    print("\n--- Evaluation on Holdout Test Set ---")
    print(clf_report)
    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f} | F1 Macro: {f1:.4f}")

    # 6. Save Artifacts Locally
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = artifacts_dir / "intent_model.pkl"
    joblib.dump(pipeline, model_artifact_path)
    print(f"Saved model artifact to: {model_artifact_path}")

    # Save metrics and classification report
    report_file = artifacts_dir / "classification_report.txt"
    report_file.write_text(clf_report, encoding="utf-8")

    metrics_summary = {
        "author": author,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": num_samples,
        "intents": unique_intents,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }

    metrics_file = artifacts_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to: {metrics_file}")

    # 7. MLflow Experiment Tracking
    if MLFLOW_AVAILABLE:
        try:
            print("\nLogging run to MLflow...")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"run_by_{author}"):
                mlflow.set_tags({
                    "author": author,
                    "model_type": "TF-IDF + LogisticRegression",
                    "framework": "scikit-learn",
                })
                # Log hyperparameters
                for k, v in params.items():
                    mlflow.log_param(k, str(v))
                mlflow.log_param("num_classes", len(unique_intents))
                mlflow.log_param("total_samples", num_samples)

                # Log evaluation metrics
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("precision_macro", float(prec))
                mlflow.log_metric("recall_macro", float(rec))
                mlflow.log_metric("f1_macro", float(f1))
                mlflow.log_metric("f1_weighted", float(f1_weighted))

                # Log artifacts
                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(metrics_file))
                # Log model artifact (supports MLflow 3.x and earlier)
                try:
                    mlflow.sklearn.log_model(pipeline, name="model")
                except TypeError:
                    mlflow.sklearn.log_model(pipeline, "model")
                print("MLflow tracking successfully completed.")
        except Exception as e:
            print(f"Notice: MLflow logging encountered an issue: {e}")
    else:
        print("MLflow is not installed in current environment; skipping MLflow logging.")

    print("\nTraining workflow completed successfully!")
    return metrics_summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "intents.csv"
    artifacts_dir = base_dir / "model" / "artifacts"
    train_model(data_path, artifacts_dir)
