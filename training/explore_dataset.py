from pathlib import Path
import pandas as pd

for name in ["LID_train.csv", "LID_test.csv"]:
    path = Path("data/raw") / name
    df = pd.read_csv(path)
    print("\n" + "="*70)
    print(name)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst rows:")
    print(df.head().to_string())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nValue counts:")
    for col in df.columns:
        if df[col].nunique() <= 20:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False))
