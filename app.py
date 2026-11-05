
"""
Streamlit Front-end for ML Assignment 2
Evaluate saved classifiers on an uploaded test CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# --------------------------------------------------
# Load Metadata and Saved Models
# --------------------------------------------------

@st.cache_data
def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_pipeline(model_key: str):
    filename = MODEL_FILES[model_key]
    model_path = MODEL_DIR / filename

    return joblib.load(model_path)


# --------------------------------------------------
# Prediction Scores for AUC
# --------------------------------------------------

def get_prediction_scores(pipeline, X):
    """
    Return probability or decision scores for AUC calculation.
    Positive class is spam (1).
    """

    if not hasattr(pipeline, "classes_"):
        raise ValueError(
            "Model does not expose class labels."
        )

    classes = np.asarray(pipeline.classes_)

    if len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError(
            "AUC calculation requires binary classifier 0 and 1."
        )

    positive_index = np.where(classes == 1)[0][0]

    if hasattr(pipeline, "predict_proba"):

        probabilities = pipeline.predict_proba(X)

        if probabilities.shape[1] != 2:
            raise ValueError(
                "Unexpected probability output shape."
            )

        return probabilities[:, positive_index]

    if hasattr(pipeline, "decision_function"):

        scores = pipeline.decision_function(X)

        if np.asarray(scores).ndim != 1:
            raise ValueError(
                "Expected one-dimensional binary decision scores."
            )

        # Binary decision scores are oriented toward classes_[1].
        return scores if positive_index == 1 else -scores

    raise ValueError(
        "Selected model does not provide probability "
        "or decision scores for AUC calculation."
    )

# --------------------------------------------------
# Calculate Evaluation Metrics
# --------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray
) -> dict[str, float]:

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------
# Validate Uploaded CSV
# --------------------------------------------------

def validate_uploaded_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str
) -> pd.DataFrame:

    required_cols = feature_cols + [target_col]

    # Check empty dataset
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    # Check duplicate columns
    if df.columns.duplicated().any():
        raise ValueError(
            "Uploaded CSV contains duplicate column names."
        )

    # Check required columns
    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Select required columns only
    data = df[required_cols].copy()

    # Convert features to numeric
    for col in feature_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Check missing or invalid feature values
    if data[feature_cols].isna().any().any():
        raise ValueError(
            "Features contain missing or non-numeric values. "
            "Please correct the uploaded CSV."
        )

    # Check infinite feature values
    if not np.isfinite(
        data[feature_cols].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Features contain infinite values."
        )

    # Convert target column to numeric
    data[target_col] = pd.to_numeric(
        data[target_col],
        errors="coerce"
    )

    # Check missing target values
    if data[target_col].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    # Check infinite target values
    if not np.isfinite(
            data[target_col].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Target column contains infinite values."
        )

    # Check target values
    if not set(data[target_col].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    # Check both classes are present for AUC
    if data[target_col].nunique() < 2:
        raise ValueError(
            "AUC cannot be calculated because the test data "
            "must contain both classes: ham (0) and spam (1)."
        )

    return data


# --------------------------------------------------
# Evaluate All Five Models
# --------------------------------------------------

def evaluate_all_models(X, y_true):

    comparison_results = []
    predictions = {}

    for model_name in MODEL_FILES:

        pipeline = load_pipeline(model_name)

        y_pred = pipeline.predict(X)

        y_score = get_prediction_scores(
            pipeline, X
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
            y_score
        )

        comparison_results.append({
            "Model": model_name,
            **metrics
        })

        predictions[model_name] = y_pred

    comparison_df = pd.DataFrame(
        comparison_results
    )

    return comparison_df, predictions


# --------------------------------------------------
# Main Streamlit Application
# --------------------------------------------------

def main() -> None:

    st.set_page_config(
        page_title="Spam Email Classifier — ML Assignment 2",
        page_icon="📧",
        layout="wide",
    )
    
  
    # Load and validate metadata
    try:
        meta = load_metadata()

        target_col = meta["target_column"]
        feature_cols = meta["feature_columns"]

        # Validate target column
        if not isinstance(target_col, str) or not target_col.strip():
            raise ValueError(
                "Invalid target_column in metadata."
            )

        # Validate feature columns
        if (
            not isinstance(feature_cols, list)
            or not feature_cols
            or not all(
                isinstance(col, str) and col.strip()
                for col in feature_cols
            )
        ):
            raise ValueError(
                "Invalid feature_columns in metadata."
            )

        # Check duplicate feature names
        if len(feature_cols) != len(set(feature_cols)):
            raise ValueError(
                "Duplicate feature names found in metadata."
            )

        # Target must not be a feature
        if target_col in feature_cols:
            raise ValueError(
                "Target column must not be included in feature_columns."
            )

    except Exception as exc:
        st.error(
            f"Could not load or validate model metadata: {exc}"
        )
        return

    # Application title
    st.title("Email Spam Classification Dashboard")    
    
    st.markdown(
        "Upload a **test CSV** containing the features and target. "
        "The application loads pre-trained models and evaluates "
        "the uploaded data without retraining."
    )

    col_left, col_right = st.columns([1, 2])

    # --------------------------------------------------
    # Configuration Section
    # --------------------------------------------------

    with col_left:

        st.subheader("Configuration")

        model_options = ["All Models"] + list(MODEL_FILES.keys())

        model_name = st.selectbox(
	    "Select classifier",
	    model_options
	)

        uploaded = st.file_uploader(
            "Upload test CSV",
            type=["csv"]
        )


        st.info(
            f"Expected target column: `{target_col}` · "
            f"{len(feature_cols)} numeric features"
        )

    # Check file upload
    if uploaded is None:

        st.warning(
            "Upload `test_data.csv` from the repository "
            "to see evaluation metrics."
        )

        return

    # --------------------------------------------------
    # Read Uploaded CSV
    # --------------------------------------------------

    try:

        df = pd.read_csv(uploaded)

    except Exception as exc:

        st.error(f"Could not read CSV: {exc}")
        return

    # --------------------------------------------------
    # Validate Uploaded CSV
    # --------------------------------------------------

    try:

        df = validate_uploaded_data(
            df,
            feature_cols,
            target_col
        )

    except ValueError as exc:

        st.error(str(exc))
        return

    # --------------------------------------------------
    # Prepare Features and Target
    # --------------------------------------------------

    X = df[feature_cols]

    y_true = df[target_col].astype(int).values

    # --------------------------------------------------
    # Uploaded Dataset Information
    # --------------------------------------------------

    st.subheader("Uploaded Dataset Information")

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Total Records",
        len(df)
    )

    info2.metric(
        "Input Features",
        len(feature_cols)
    )

    info3.metric(
        "Spam Percentage",
        f"{y_true.mean() * 100:.2f}%"
    )

    # --------------------------------------------------
    # Evaluate All Five Classifiers
    # --------------------------------------------------

    st.subheader("Comparison of All Five Classifiers")

    try:

        comparison_df, all_predictions = (
            evaluate_all_models(X, y_true)
        )

        # Display comparison table
        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Download comparison results
        csv_output = comparison_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Model Comparison CSV",
            data=csv_output,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )

    except Exception as exc:

        st.error(
            f"Model comparison failed: {exc}"
        )

        return


    # --------------------------------------------------
    # Display Model Results
    # --------------------------------------------------

    with col_right:

        if model_name == "All Models":

            st.subheader("Detailed Results — All Five Models")

            for current_model in MODEL_FILES:

                y_pred = all_predictions[current_model]

                selected_result = comparison_df.loc[
                    comparison_df["Model"] == current_model
                ]

                metrics = selected_result.iloc[0].drop(
                    "Model"
                ).to_dict()

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                )

                with st.expander(
                    f"Results — {current_model}",
                    expanded=False
                ):

                    # Display evaluation metrics
                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Accuracy",
                        f"{metrics['Accuracy']:.4f}"
                    )

                    m2.metric(
                        "AUC",
                        f"{metrics['AUC']:.4f}"
                    )

                    m3.metric(
                        "F1 Score",
                        f"{metrics['F1 Score']:.4f}"
                    )

                    m4, m5, m6 = st.columns(3)

                    m4.metric(
                        "Precision",
                        f"{metrics['Precision']:.4f}"
                    )

                    m5.metric(
                        "Recall",
                        f"{metrics['Recall']:.4f}"
                    )

                    m6.metric(
                        "MCC",
                        f"{metrics['MCC']:.4f}"
                    )

                    # Display confusion matrix
                    st.write("Confusion Matrix")

                    cm_df = pd.DataFrame(
                        cm,
                        index=[
                            "Actual: ham (0)",
                            "Actual: spam (1)"
                        ],
                        columns=[
                            "Pred: ham (0)",
                            "Pred: spam (1)"
                        ]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    sns.heatmap(
                        cm_df,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        linewidths=0.5,
                        ax=ax
                    )

                    ax.set_title(
                        f"Confusion Matrix - {current_model}"
                    )

                    ax.set_xlabel("Predicted Class")
                    ax.set_ylabel("Actual Class")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Display classification report
                    st.write("Classification Report")

                    report = classification_report(
                        y_true,
                        y_pred,
                        labels=[0, 1],
                        target_names=[
                            "Ham (Legitimate Email)",
                            "Spam"
                        ],
                        digits=4,
                        zero_division=0,
                        output_dict=True
                    )

                    report_df = pd.DataFrame(
                        report
                    ).transpose()

                    st.dataframe(
                        report_df.style.format("{:.4f}"),
                        use_container_width=True
                    )

        else:

            y_pred = all_predictions[model_name]

            selected_result = comparison_df.loc[
                comparison_df["Model"] == model_name
            ]

            metrics = selected_result.iloc[0].drop(
                "Model"
            ).to_dict()

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            st.subheader(f"Results — {model_name}")

            m1, m2, m3 = st.columns(3)

            m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            m2.metric("AUC", f"{metrics['AUC']:.4f}")
            m3.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

            m4, m5, m6 = st.columns(3)

            m4.metric("Precision", f"{metrics['Precision']:.4f}")
            m5.metric("Recall", f"{metrics['Recall']:.4f}")
            m6.metric("MCC", f"{metrics['MCC']:.4f}")

            tab_cm, tab_report = st.tabs([
                "Confusion Matrix",
                "Classification Report"
            ])

            with tab_cm:

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual: ham (0)",
                        "Actual: spam (1)"
                    ],
                    columns=[
                        "Pred: ham (0)",
                        "Pred: spam (1)"
                    ]
                )

                fig, ax = plt.subplots(figsize=(6, 4))

                sns.heatmap(
                    cm_df,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    linewidths=0.5,
                    ax=ax
                )

                ax.set_title(
                    f"Confusion Matrix - {model_name}"
                )

                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Actual Class")

                st.pyplot(fig)
                plt.close(fig)

            with tab_report:

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=[
                        "Ham (Legitimate Email)",
                        "Spam"
                    ],
                    digits=4,
                    zero_division=0,
                    output_dict=True
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df.style.format("{:.4f}"),
                    use_container_width=True
                )

    # --------------------------------------------------
    # Application Footer
    # --------------------------------------------------

    st.caption(
        f"Evaluated {len(df)} rows · "
        "Models trained offline — no retraining on upload."
    )

    # Evaluate Selected Model
    # --------------------------------------------------  
    try:

        y_pred = all_predictions[model_name]

        selected_result = comparison_df.loc[
        comparison_df["Model"] == model_name
        ]

        metrics = selected_result.iloc[0].drop(
            "Model"
        ).to_dict()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

    except Exception as exc:

        st.error(
            f"Selected model evaluation failed: {exc}"
        )
        return


    # --------------------------------------------------
    # Display Selected Model Results
    # --------------------------------------------------

    with col_right:

        st.subheader(f"Results — {model_name}")

        # First row of metrics
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.4f}"
        )

        m2.metric(
            "AUC",
            f"{metrics['AUC']:.4f}"
        )

        m3.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.4f}"
        )

        # Second row of metrics
        m4, m5, m6 = st.columns(3)

        m4.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

        m5.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

        m6.metric(
            "MCC",
            f"{metrics['MCC']:.4f}"
        )

        # Tabs
        tab_cm, tab_report = st.tabs([
            "Confusion Matrix",
            "Classification Report"
        ])

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        with tab_cm:

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual: ham (0)",
                    "Actual: spam (1)"
                ],
                columns=[
                    "Pred: ham (0)",
                    "Pred: spam (1)"
                ]
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            sns.heatmap(
                cm_df,
                annot=True,
                fmt="d",
                cmap="Blues",
                linewidths=0.5,
                ax=ax
            )

            ax.set_title(
                f"Confusion Matrix - {model_name}"
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            st.pyplot(fig)

            plt.close(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        with tab_report:

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "Ham (Legitimate Email)",
                    "Spam"
                ],
                digits=4,
                zero_division=0,
                output_dict=True
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.style.format("{:.4f}"),
                use_container_width=True
            )

    # --------------------------------------------------
    # Application Footer
    # --------------------------------------------------

    st.caption(
        f"Evaluated {len(df)} rows · "
        "Models trained offline — no retraining on upload."
    )

if __name__ == "__main__":
    main()


