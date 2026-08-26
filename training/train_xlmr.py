from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TRAIN_FILE = Path(
    "data/processed/LID_train_processed.csv"
)

TEST_FILE = Path(
    "data/processed/LID_test_processed.csv"
)

MODEL_DIR = Path(
    "models/saved/xlmr"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

# First experiment:
# 30,000 samples per class
#
# Total:
# Hindi    = 30,000
# English  = 30,000
# Other    = 30,000
# ----------------
# Total    = 90,000

TRAIN_PER_CLASS = 30000

MAX_LENGTH = 64

LABELS = [
    "hi",
    "en",
    "ot",
]


LABEL2ID = {
    "hi": 0,
    "en": 1,
    "ot": 2,
}


ID2LABEL = {
    0: "hi",
    1: "en",
    2: "ot",
}


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("XLM-RoBERTa TRAINING")
print("=" * 70)

print("\nPyTorch:", torch.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "\nCUDA GPU is not available.\n"
        "Please verify your PyTorch CUDA installation."
    )


GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / (1024 ** 3)
)


print(
    "GPU:",
    GPU_NAME
)

print(
    "GPU memory:",
    f"{GPU_MEMORY:.2f} GB"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    "\nFull training size:",
    f"{len(train_df):,}"
)

print(
    "Test size:",
    f"{len(test_df):,}"
)


# ============================================================
# VERIFY LABELS
# ============================================================

print("\nTraining labels:")

print(
    train_df["label"].value_counts()
)


print("\nTest labels:")

print(
    test_df["label"].value_counts()
)


invalid_train = (
    set(train_df["label"].unique())
    - set(LABELS)
)

invalid_test = (
    set(test_df["label"].unique())
    - set(LABELS)
)


if invalid_train:

    raise ValueError(
        f"Invalid training labels: "
        f"{invalid_train}"
    )


if invalid_test:

    raise ValueError(
        f"Invalid test labels: "
        f"{invalid_test}"
    )


# ============================================================
# CLEAN TOKEN DATA
# ============================================================

print("\nCleaning token data...")


train_df["token"] = (
    train_df["token"]
    .fillna("")
    .astype(str)
    .str.strip()
)


test_df["token"] = (
    test_df["token"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty tokens

train_df = train_df[
    train_df["token"] != ""
].reset_index(drop=True)


test_df = test_df[
    test_df["token"] != ""
].reset_index(drop=True)


print(
    "Training samples after cleaning:",
    f"{len(train_df):,}"
)

print(
    "Test samples after cleaning:",
    f"{len(test_df):,}"
)


# ============================================================
# BALANCED TRAINING SUBSET
# ============================================================

print("\n" + "=" * 70)
print("CREATING BALANCED TRAINING DATA")
print("=" * 70)


parts = []


for label in LABELS:

    subset = train_df[
        train_df["label"] == label
    ].copy()


    available = len(subset)

    sample_size = min(
        TRAIN_PER_CLASS,
        available
    )


    print(
        f"{label}: "
        f"available={available:,}, "
        f"selected={sample_size:,}"
    )


    sampled = subset.sample(
        n=sample_size,
        random_state=42
    )


    parts.append(
        sampled
    )


train_df = pd.concat(
    parts,
    ignore_index=True
)


# Shuffle

train_df = train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print(
    "\nBalanced training size:",
    f"{len(train_df):,}"
)


print("\nBalanced distribution:")

print(
    train_df["label"].value_counts()
)


# ============================================================
# CREATE LABEL IDs
# ============================================================

train_df["label_id"] = (
    train_df["label"]
    .map(LABEL2ID)
)


test_df["label_id"] = (
    test_df["label"]
    .map(LABEL2ID)
)


# ============================================================
# CREATE HUGGING FACE DATASETS
# ============================================================

print("\nCreating Hugging Face datasets...")


train_dataset = Dataset.from_pandas(
    train_df[
        [
            "token",
            "label_id",
        ]
    ],
    preserve_index=False
)


test_dataset = Dataset.from_pandas(
    test_df[
        [
            "token",
            "label_id",
        ]
    ],
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING XLM-R TOKENIZER")
print("=" * 70)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION FUNCTION
# ============================================================

def tokenize_function(examples):

    # Make absolutely sure every input is a string.

    texts = [
        str(text)
        for text in examples["token"]
    ]


    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,

        # Needed so we can ignore special tokens
        # such as <s> and </s>.
        return_special_tokens_mask=True,
    )


    labels = []


    for label_id, special_mask in zip(
        examples["label_id"],
        tokenized["special_tokens_mask"]
    ):

        token_labels = []


        for is_special in special_mask:

            if is_special == 1:

                # Ignore special tokens during loss
                # and evaluation.
                token_labels.append(-100)

            else:

                # Assign original language label
                # to XLM-R subword token.
                token_labels.append(
                    int(label_id)
                )


        labels.append(
            token_labels
        )


    tokenized["labels"] = labels


    return tokenized


# ============================================================
# TOKENIZE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING TRAINING DATA")
print("=" * 70)


train_tokenized = train_dataset.map(
    tokenize_function,

    batched=True,

    remove_columns=[
        "token",
        "label_id",
    ],

    desc="Tokenizing train",
)


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING TEST DATA")
print("=" * 70)


test_tokenized = test_dataset.map(
    tokenize_function,

    batched=True,

    remove_columns=[
        "token",
        "label_id",
    ],

    desc="Tokenizing test",
)


# ============================================================
# LOAD XLM-R MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING XLM-RoBERTa MODEL")
print("=" * 70)


model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,

    num_labels=3,

    id2label=ID2LABEL,

    label2id=LABEL2ID,
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction


    # Convert logits → predicted class

    predictions = np.argmax(
        predictions,
        axis=2
    )


    true_predictions = []

    true_labels = []


    for prediction_row, label_row in zip(
        predictions,
        labels
    ):

        for prediction, label in zip(
            prediction_row,
            label_row
        ):

            # Ignore special tokens
            # marked as -100.

            if label != -100:

                true_predictions.append(
                    int(prediction)
                )

                true_labels.append(
                    int(label)
                )


    accuracy = accuracy_score(
        true_labels,
        true_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            true_predictions,

            average="macro",

            zero_division=0,
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("\nPreparing training configuration...")


training_args = TrainingArguments(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=str(
        MODEL_DIR / "checkpoints"
    ),


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    save_strategy="epoch",


    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=2e-5,


    # --------------------------------------------------------
    # GPU batch sizes
    # --------------------------------------------------------

    # RTX 4050 ~6 GB VRAM
    per_device_train_batch_size=8,

    per_device_eval_batch_size=16,


    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    gradient_accumulation_steps=2,


    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    num_train_epochs=2,


    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    weight_decay=0.01,


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=100,


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_total_limit=2,


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,


    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    # RTX 4050 supports FP16.
    fp16=True,


    # --------------------------------------------------------
    # Disable external logging
    # --------------------------------------------------------

    report_to="none",


    # --------------------------------------------------------
    # Windows stability
    # --------------------------------------------------------

    dataloader_num_workers=0,
)


# ============================================================
# CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=test_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING XLM-R TRAINING")
print("=" * 70)

print(
    "\nTraining on:",
    GPU_NAME
)

print(
    "Training samples:",
    f"{len(train_df):,}"
)

print(
    "Test samples:",
    f"{len(test_df):,}"
)

print(
    "Epochs:",
    2
)

print(
    "Batch size:",
    8
)

print(
    "Gradient accumulation:",
    2
)

print(
    "\nPlease wait..."
)


trainer.train()


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)


metrics = trainer.evaluate()


for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)


trainer.save_model(
    str(MODEL_DIR)
)


tokenizer.save_pretrained(
    str(MODEL_DIR)
)


print(
    "\nModel saved to:"
)

print(
    MODEL_DIR
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("XLM-R TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)