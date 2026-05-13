from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train six models on the Iris dataset and save artifacts and outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--data-path",
        default=str(root / "data" / "dataset.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(root / "outputs"),
        help="Directory for output JSON files.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(root / "models"),
        help="Directory for model artifacts.",
    )
    parser.add_argument(
        "--target-column",
        default="Species",
        help="Target column name.",
    )
    parser.add_argument(
        "--id-column",
        default="Id",
        help="Optional ID column to drop if present.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    return parser.parse_args()


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def round_report(report: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            rounded[key] = {}
            for metric, metric_value in value.items():
                if isinstance(metric_value, (np.floating, float)):
                    rounded[key][metric] = round_float(metric_value)
                elif isinstance(metric_value, (np.integer, int)):
                    rounded[key][metric] = int(metric_value)
                else:
                    rounded[key][metric] = metric_value
        else:
            if isinstance(value, (np.floating, float)):
                rounded[key] = round_float(value)
            elif isinstance(value, (np.integer, int)):
                rounded[key] = int(value)
            else:
                rounded[key] = value
    return rounded


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_models(random_state: int) -> list[tuple[str, Pipeline, str]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000, random_state=random_state, multi_class="auto"
                        ),
                    ),
                ]
            ),
            "model_logistic_regression.joblib",
        ),
        (
            "svc_rbf",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", probability=True, random_state=random_state)),
                ]
            ),
            "model_svc_rbf.joblib",
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200, random_state=random_state
                        ),
                    )
                ]
            ),
            "model_random_forest.joblib",
        ),
        (
            "gradient_boosting",
            Pipeline(
                steps=[
                    (
                        "model",
                        GradientBoostingClassifier(random_state=random_state),
                    )
                ]
            ),
            "model_gradient_boosting.joblib",
        ),
        (
            "knn",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
            "model_knn.joblib",
        ),
        (
            "decision_tree",
            Pipeline(
                steps=[
                    ("model", DecisionTreeClassifier(random_state=random_state))
                ]
            ),
            "model_decision_tree.joblib",
        ),
    ]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    data_path = Path(args.data_path)
    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.models_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if args.target_column not in df.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found in dataset columns."
        )

    if args.id_column in df.columns:
        df = df.drop(columns=[args.id_column])

    feature_columns = [col for col in df.columns if col != args.target_column]
    if not feature_columns:
        raise ValueError("No feature columns found after preprocessing.")

    X = df[feature_columns]
    y = df[args.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_splits, shuffle=True, random_state=args.random_state
    )

    models = build_models(args.random_state)

    model_results: list[dict[str, Any]] = []
    best_model_name = ""
    best_model_metrics = (-1.0, -1.0)
    best_model_eval: dict[str, Any] = {}
    best_model_artifact = ""

    print(f"Training 6 models on: {data_path}")

    for model_name, pipeline, artifact_name in models:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")

        artifact_path = models_dir / artifact_name
        pipeline.fit(X, y)
        dump(pipeline, artifact_path)

        result = {
            "name": model_name,
            "artifact_path": safe_relative_path(artifact_path, root),
            "cv_accuracy_mean": round_float(cv_scores.mean()),
            "cv_accuracy_std": round_float(cv_scores.std()),
            "test_accuracy": round_float(test_accuracy),
            "test_f1_macro": round_float(test_f1_macro),
        }
        model_results.append(result)

        if (test_accuracy, test_f1_macro) > best_model_metrics:
            best_model_metrics = (test_accuracy, test_f1_macro)
            best_model_name = model_name
            best_model_artifact = result["artifact_path"]
            class_labels = sorted(y.unique())
            report = classification_report(
                y_test, y_pred, output_dict=True, labels=class_labels
            )
            best_model_eval = {
                "accuracy": round_float(test_accuracy),
                "f1_macro": round_float(test_f1_macro),
                "confusion_matrix": confusion_matrix(
                    y_test, y_pred, labels=class_labels
                ).tolist(),
                "classification_report": round_report(report),
                "class_labels": class_labels,
            }

        print(
            f"Saved {model_name} -> {artifact_path} | test_accuracy={result['test_accuracy']}"
        )

    model_results_sorted = sorted(
        model_results, key=lambda item: item["test_accuracy"], reverse=True
    )

    day4_output = {
        "status": "DONE",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": {
            "path": safe_relative_path(data_path, root),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_column": args.target_column,
            "feature_columns": feature_columns,
        },
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratify": True,
        },
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": args.cv_splits,
            "shuffle": True,
            "random_state": args.random_state,
        },
        "models": model_results_sorted,
        "best_model": {
            "name": best_model_name,
            "selection_metric": "test_accuracy",
            "artifact_path": best_model_artifact,
        },
    }

    day5_output = {
        "status": "DONE",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "best_model": {
            "name": best_model_name,
            "artifact_path": best_model_artifact,
        },
        "test_metrics": best_model_eval,
    }

    day4_path = outputs_dir / "day4_models.json"
    day5_path = outputs_dir / "day5_eval.json"

    with day4_path.open("w", encoding="utf-8") as file:
        json.dump(day4_output, file, indent=2)

    with day5_path.open("w", encoding="utf-8") as file:
        json.dump(day5_output, file, indent=2)

    print(f"Wrote outputs -> {day4_path} and {day5_path}")


if __name__ == "__main__":
    main()
