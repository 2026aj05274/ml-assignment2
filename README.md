# ML Assignment 2 — Classification & Streamlit Deployment

**Programme:** M.Tech (AIML) · **Course:** Machine Learning · **Marks:** 15 · **Deadline:** 08-Nov-2026

End-to-end binary classification on the UCI Spambase dataset: five scikit-learn classifiers, held-out test evaluation, saved pipelines, and a Streamlit app that scores uploaded CSVs without retraining.

---

## a. Problem statement

Email spam wastes storage and bandwidth and is often used to spread malware or phishing links. The goal is to **classify each email as spam (1) or legitimate ham (0)** from measurable text and layout statistics (word frequencies, character frequencies, capital-letter run lengths).

This project trains five standard classifiers on the same labeled data, compares them with accuracy, AUC, precision, recall, F1, and MCC on a fixed test split, persists the fitted pipelines, and exposes them through an interactive Streamlit dashboard for evaluators.

---

## b. Dataset description

| Item | Detail |
|------|--------|
| **Name** | Spambase (UCI Machine Learning Repository) |
| **Source** | [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/spambase) |
| **Task** | Binary classification (`is_spam`: 0 = ham, 1 = spam) |
| **Instances** | 4,601 emails |
| **Features** | 57 numeric inputs (48 word-frequency percentages, 6 character-frequency percentages, 3 capital-run-length measures) |
| **Missing values** | None in the raw UCI file |
| **Split** | 75% train / 25% stratified test (`random_state=42`); **`test_data.csv`** in this repo is the held-out 1,151 rows (features + `is_spam`) |

Features are continuous proportions in \([0, 100]\); The dataset contains numerical features representing word frequencies, character frequencies, and capital-letter run lengths. The preprocessing pipeline applies median imputation to the input features. StandardScaler is applied before Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. Decision Tree uses imputation without feature scaling.

---
## c. Data preprocessing and methodology

The UCI Spambase dataset contains 4,601 instances and 57 numerical input features. The target variable is `is_spam`, where 0 represents a legitimate email (ham) and 1 represents a spam email.

The dataset is divided into training and testing sets using a 75:25 stratified split with random_state=42.

The following preprocessing steps are applied:

1. Median imputation is used to handle missing numerical values.
2. StandardScaler is applied for Logistic Regression, KNN, Gaussian Naive Bayes, and Support Vector Machine.
3. Decision Tree uses median imputation without feature scaling.

All five classifiers are trained using the same training dataset and evaluated on the same held-out test dataset.

The trained preprocessing pipelines and classifiers are saved as Joblib files in the `model/` directory. The Streamlit application loads these saved models and evaluates uploaded test data without retraining.


## d. GitHub repository link

**Replace with your repository URL after push:**

`https://github.com/2026aj05274/ml-assignment2`

**Repository layout:**

```
ml-assignment2/
├── app.py                 # Streamlit UI
├── train_models.py        # Training & model export├── requirements.txt
├── README.md
├── test_data.csv          # Held-out test set (upload this in the app)
├── metrics.json           # Test metrics used in the table below
├── data/spambase_full.csv # Full dataset snapshot
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

---

## e. Models used

All models share the same preprocessing and test set. Metrics below are on **`test_data.csv`** (no retraining at inference time).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9279 | 0.9728 | 0.9304 | 0.8833 | 0.9062 | 0.8485 |
| Decision Tree | 0.9235 | 0.9162 | 0.9049 | 0.9009 | 0.9029 | 0.8398 |
| KNN | 0.9157 | 0.9621 | 0.8993 | 0.8855 | 0.8923 | 0.8232 |
| Naive Bayes | 0.8254 | 0.9368 | 0.7077 | 0.9493 | 0.8109 | 0.6795 |
| Support Vector Machine | 0.9288 | 0.9701 | 0.9326 | 0.8833 | 0.9072 | 0.8503 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|----------------------------------------|
| **Logistic Regression** | Strong overall balance: high AUC (~0.97) and MCC with relatively few false positives. Linear decision boundary works well because many spam cues appear as monotonic frequency shifts. |
| **Decision Tree** | Accuracy close to logistic regression but **lower AUC**, suggesting weaker probability ranking despite good hard labels. Slightly higher recall on spam than logistic regression, with more false positives on ham. |
| **KNN** | Distance-based voting captures local structure; performance is solid but slightly below linear/SVM models on accuracy and MCC. Sensitive to scaling (addressed here with `StandardScaler`). |
| **Naive Bayes** | **Lowest accuracy and precision**; many ham emails misclassified as spam (high recall ~0.95 but precision ~0.71). Gaussian independence assumptions are a poor fit for correlated word-frequency features. |
| **Support Vector Machine** | **Best overall on this split**: highest accuracy, F1, and MCC, with AUC comparable to logistic regression. RBF kernel models non-linear boundaries between spam and ham clusters. |
| **Overall, winner for your dataset?** | **Support Vector Machine** (marginally ahead of Logistic Regression on accuracy/F1/MCC). For deployment, logistic regression remains an interpretable alternative with nearly identical AUC. |

---

## Streamlit app (local run)

```bash
cd ml-assignment2-classification
pip install -r requirements.txt
python train_models.py          # only if model/ is missing
python -m streamlit run app.py
```

In the app: upload `test_data.csv`, pick a model from the dropdown, and view metrics plus confusion matrix / classification report.

### Live deployment link

**After [Streamlit Community Cloud](https://streamlit.io/cloud) deploy, paste your app URL here:**

`https://<your-app>.streamlit.app`

Deploy settings: repository = this project, branch = `main`, main file = **`app.py`**.

---

## Reproducing training (BITS Virtual Lab)

1. Open the project folder in BITS Virtual Lab.
2. Run `pip install -r requirements.txt` then `python train_models.py`.
3. Run `python -m streamlit run app.py` and upload `test_data.csv`.
4. Capture **one screenshot** of execution for the PDF submission.

---

## Academic integrity

Original implementation for this assignment; dataset and problem are public UCI benchmarks. Commit history reflects your own work before submission.
