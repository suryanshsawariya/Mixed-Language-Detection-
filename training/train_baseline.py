from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/LID_train_processed.csv"
)

TEST_FILE = Path(
    "data/processed/LID_test_processed.csv"
)

MODEL_DIR = Path("models/saved")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training samples: {len(train_df):,}")
print(f"Testing samples:  {len(test_df):,}")


# ============================================================
# FEATURES / LABELS
# ============================================================

X_train = train_df["token"].astype(str)
y_train = train_df["label"].astype(str)

X_test = test_df["token"].astype(str)
y_test = test_df["label"].astype(str)


# ============================================================
# MODEL
# ============================================================

print("\nCreating TF-IDF + Logistic Regression...")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")
print("Please wait...")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed!")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("BASELINE MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["hi", "en", "ot"],
        target_names=[
            "Hindi",
            "English",
            "Other",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["hi", "en", "ot"],
)

print("\nConfusion Matrix:")
print(
    "Rows = Actual"
)
print(
    "Columns = Predicted"
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "baseline_logistic_regression.joblib"
)

joblib.dump(
    model,
    model_path
)

print(
    f"\nModel saved to:"
)
print(model_path)


print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETED")
print("=" * 70)