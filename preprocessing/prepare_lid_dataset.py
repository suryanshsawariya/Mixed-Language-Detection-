import ast
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANNOTATOR COLUMNS
# ============================================================

ANNOTATOR_COLUMNS = [
    "Annotated by: Annotator 1",
    "Annotated by: Annotator 2",
    "Annotated by: Annotator 3",
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):
    """
    Convert COMI-LINGUA raw labels into 3 final classes.

    Raw labels:
        hi = Hindi
        h  = Romanized Hindi
        en = English
        e  = English
        ot = Other
        u  = Other / special token

    Final labels:
        hi
        en
        ot
    """

    if label is None:
        return "ot"

    label = str(label).strip().lower()

    mapping = {
        "hi": "hi",
        "h": "hi",

        "en": "en",
        "e": "en",

        "ot": "ot",
        "u": "ot",
    }

    return mapping.get(label, "ot")


# ============================================================
# PARSE ANNOTATION
# ============================================================

def parse_tags(value):
    """
    Convert annotation string into a Python object.
    """

    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)

    except (ValueError, SyntaxError, TypeError):
        return []


# ============================================================
# FLATTEN NESTED ANNOTATIONS
# ============================================================

def flatten_annotations(obj):
    """
    Handle both normal and nested COMI-LINGUA structures.

    Example 1:
        [
            {'key': 'word', 'value': 'hi', 'index': 0}
        ]

    Example 2:
        [
            [
                {'key': 'word', 'value': 'hi', 'index': 0}
            ]
        ]
    """

    # Dictionary
    if isinstance(obj, dict):

        if "key" in obj and "value" in obj:
            return [obj]

        return []


    # List
    if isinstance(obj, list):

        result = []

        for item in obj:

            result.extend(
                flatten_annotations(item)
            )

        return result


    return []


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_label(labels):
    """
    Normalize labels and perform majority voting.

    Example:

        ['hi', 'hi', 'en']
            ↓
        ['hi', 'hi', 'en']
            ↓
        hi

    Example:

        ['h', 'hi', 'hi']
            ↓
        ['hi', 'hi', 'hi']
            ↓
        hi
    """

    normalized_labels = [
        normalize_label(label)
        for label in labels
    ]

    if not normalized_labels:
        return "ot"

    counts = Counter(normalized_labels)

    label, count = counts.most_common(1)[0]

    # Require agreement from at least 2 annotators
    if count >= 2:
        return label

    return "ot"


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filename):

    input_path = RAW_DIR / filename

    output_path = (
        PROCESSED_DIR
        / filename.replace(
            ".csv",
            "_processed.csv"
        )
    )

    print("\n" + "=" * 70)

    print("Reading:")
    print(input_path)

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    print("\nOriginal shape:")
    print(df.shape)


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "Sentences",
        "Predicted tags",
        *ANNOTATOR_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


    # --------------------------------------------------------
    # RECORDS
    # --------------------------------------------------------

    records = []


    # ========================================================
    # PROCESS EACH SENTENCE
    # ========================================================

    for row_number, row in df.iterrows():

        all_annotations = []


        # ----------------------------------------------------
        # PARSE THREE ANNOTATORS
        # ----------------------------------------------------

        for column in ANNOTATOR_COLUMNS:

            raw_value = row[column]

            parsed = parse_tags(
                raw_value
            )

            annotations = flatten_annotations(
                parsed
            )

            all_annotations.append(
                annotations
            )


        # ----------------------------------------------------
        # TOKEN COUNT
        # ----------------------------------------------------

        # Use Annotator 1 as the primary token sequence.
        #
        # The annotators normally contain the same token
        # sequence. We therefore use Annotator 1 to avoid
        # accidentally creating duplicate/misaligned tokens.

        if not all_annotations:
            continue

        base_annotations = all_annotations[0]


        # ----------------------------------------------------
        # PROCESS EACH TOKEN
        # ----------------------------------------------------

        for i, token_info in enumerate(
            base_annotations
        ):

            # Safety check
            if not isinstance(
                token_info,
                dict
            ):
                continue


            token = token_info.get(
                "key",
                ""
            )


            # Skip empty tokens
            if not token:
                continue


            # ------------------------------------------------
            # GET LABEL FROM EACH ANNOTATOR
            # ------------------------------------------------

            raw_labels = []


            for annotations in all_annotations:

                if (
                    i < len(annotations)
                    and isinstance(
                        annotations[i],
                        dict
                    )
                ):

                    raw_label = annotations[i].get(
                        "value",
                        "ot"
                    )

                else:

                    raw_label = "ot"


                raw_labels.append(
                    raw_label
                )


            # ------------------------------------------------
            # NORMALIZE EACH ANNOTATOR
            # ------------------------------------------------

            normalized_labels = [
                normalize_label(label)
                for label in raw_labels
            ]


            # ------------------------------------------------
            # MAJORITY LABEL
            # ------------------------------------------------

            final_label = majority_label(
                raw_labels
            )


            # ------------------------------------------------
            # STORE RECORD
            # ------------------------------------------------

            records.append(
                {
                    "sentence": row["Sentences"],

                    "token": token,

                    "label": final_label,

                    "annotator_1": normalized_labels[0],

                    "annotator_2": normalized_labels[1],

                    "annotator_3": normalized_labels[2],
                }
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result = pd.DataFrame(
        records
    )


    # ========================================================
    # INFORMATION
    # ========================================================

    print("\nToken-level shape:")
    print(result.shape)


    print("\nFinal label distribution:")

    print(
        result["label"].value_counts(
            dropna=False
        )
    )


    print("\nUnique final labels:")

    print(
        sorted(
            result["label"].unique()
        )
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    allowed_labels = {
        "hi",
        "en",
        "ot"
    }

    actual_labels = set(
        result["label"].unique()
    )

    invalid_labels = (
        actual_labels
        - allowed_labels
    )


    if invalid_labels:

        raise ValueError(
            f"Invalid labels found: "
            f"{invalid_labels}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )


    print("\nSaved:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_file(
        "LID_train.csv"
    )

    process_file(
        "LID_test.csv"
    )

    print("\n" + "=" * 70)

    print(
        "Dataset preprocessing completed successfully."
    )