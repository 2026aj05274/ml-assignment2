"""
Train five classifiers on the UCI Spambase dataset and persist models + test split.
Run once locally (e.g. on BITS Virtual Lab) before deploying the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COL = "is_spam"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"


def load_spambase() -> pd.DataFrame:
    feature_cols = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1993",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_semicolon",
        "char_freq_left_paren",
        "char_freq_left_bracket",
        "char_freq_exclamation",
        "char_freq_dollar",
        "char_freq_hash",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
    
    local_path = DATA_DIR / "spambase_full.csv"

    if local_path.exists():
        print("Loading local Spambase dataset...")
        df = pd.read_csv(local_path)

    else:
        print("Local dataset not found. Downloading from UCI...")
        df = pd.read_csv(
             DATA_URL,
             header=None,
             names=feature_cols + [TARGET_COL]
        )
             
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)

    expected_columns = feature_cols + [TARGET_COL]

    if list(df.columns) != expected_columns:
        raise ValueError(
            "Dataset columns do not match the expected Spambase schema."
        )

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values.")

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    return df

############################
#  raw = pd.read_csv(DATA_URL, header=None, names=feature_cols + [TARGET_COL])
# return raw
###########################


def build_preprocessor(
    feature_cols: list[str],
    scale_features: bool = True
):

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_features:
        steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_cols)
        ],
        remainder="drop"
    )


#######################################################
#def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
#    numeric = Pipeline(
#        steps=[
#            ("imputer", SimpleImputer(strategy="median")),
#            ("scaler", StandardScaler()),
#        ]
#    )
#    return ColumnTransformer(
#       transformers=[("num", numeric, feature_cols)],
#        remainder="drop",
#    )
#########################################################

def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
   ############################################### 
   # if hasattr(pipeline, "predict_proba"):
   #     y_score = pipeline.predict_proba(X_test)[:, 1]
   # else:
   #     y_score = y_pred
   ###############################################
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_score)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_spambase()
    full_path = DATA_DIR / "spambase_full.csv"
    df.to_csv(full_path, index=False)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    model_specs: list[tuple[str, object]] = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        ),
        (
            "knn",
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        ),
        ("naive_bayes", GaussianNB()),
        (
            "support_vector_machine",
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        ),
    ]

    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "K-Nearest Neighbor",
        "naive_bayes": "Naive Bayes (Gaussian)",
        "support_vector_machine": "Support Vector Machine",
    }

    metrics_rows = []
    
    
   # for slug, estimator in model_specs:
    
    for slug, estimator in model_specs:

        # Decision Tree does not require scaling
        scale_features = slug != "decision_tree"

        pipeline = Pipeline(
            steps=[
                (
                    "prep",
                    build_preprocessor(
                        feature_cols,
                    scale_features=scale_features
                  )
                ),
                ("clf", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{slug}.joblib"
        )

        row = evaluate_model(
            display_names[slug],
            pipeline,
            X_test,
            y_test
        )

        metrics_rows.append(row)

        print(
            f"{row['model']}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"AUC={row['auc']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"MCC={row['mcc']:.4f}"
        )

    
    	
    
###############################################################    
#        pipeline = Pipeline(
#            steps=[
#                ("prep", build_preprocessor(feature_cols)),
#                ("clf", estimator),
#            ]
#        )
#
#        pipeline.fit(X_train, y_train)
#        joblib.dump(pipeline, MODEL_DIR / f"{slug}.joblib")
#
#        row = evaluate_model(display_names[slug], pipeline, X_test, y_test)
#        metrics_rows.append(row)
#        print(
#            f"{row['model']}: "
#            f"Acc={row['accuracy']:.4f} AUC={row['auc']:.4f} "
#            f"F1={row['f1']:.4f} MCC={row['mcc']:.4f}"
#        )
##############################################################
    meta = {
    	 "project": "UCI Spambase Classification",
    	 "target_column": TARGET_COL,
         "target_mapping": {
             "0": "Ham (Legitimate Email)",
             "1": "Spam"
        },
        "feature_columns": feature_cols,
        "random_state": RANDOM_STATE,
        "test_size_fraction": 0.25,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_instances": int(len(df)),
        "number_of_features": len(feature_cols),
        "dataset_source": "UCI Machine Learning Repository",
        "dataset_file": "data/spambase_full.csv",
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "models": list(display_names.values())
        
#######################################        
#        "target_column": TARGET_COL,
#        "feature_columns": feature_cols,
#        "random_state": RANDOM_STATE,
#        "train_size": int(len(X_train)),
#        "test_size": int(len(X_test)),
#        "dataset_source": DATA_URL,
#        "models": list(display_names.values()),
#        "sklearn_version": sklearn.__version__,
#        "pandas_version": pd.__version__,
#       "numpy_version": np.__version__,
#        "test_size_fraction": 0.25,
    }
    
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    print(f"\nSaved {len(model_specs)} models to {MODEL_DIR}")
    print(f"Test CSV: {ROOT / 'test_data.csv'} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
