# src/data_prep.py

from pathlib import Path
from typing import List
import pandas as pd


def load_livermore_qa(excel_path: str) -> pd.DataFrame:
    """
    Load Jesse Livermore Q&A data from the Excel file and return a unified DataFrame
    with columns: id, question, answer, label.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    xls = pd.ExcelFile(excel_path)

    all_rows = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        
        # Normalize column names (strip spaces etc.)
        df.columns = [str(c).strip() for c in df.columns]

        # Adjust these if your column names differ slightly
        expected_cols = {"Questions", "Answers", "Label"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"Sheet '{sheet_name}' doesn't have expected columns "
                f"{expected_cols}. Found: {list(df.columns)}"
            )

        df = df[["Questions", "Answers", "Label"]].copy()
        df.columns = ["question", "answer", "label"]

        # Clean whitespace
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()

        # Drop empty rows if any
        df = df[(df["question"] != "") & (df["answer"] != "")]
        all_rows.append(df)

    unified = pd.concat(all_rows, ignore_index=True)

    # Drop duplicates just in case
    unified = unified.drop_duplicates(
        subset=["question", "answer", "label"]
    ).reset_index(drop=True)

    # Add id column
    unified["id"] = unified.index.astype(int)

    # Reorder columns
    unified = unified[["id", "question", "answer", "label"]]

    return unified


if __name__ == "__main__":
    # Quick sanity check when you run: python -m src.data_prep
    df = load_livermore_qa("data/Team Livermore.xlsx")
    print(df.head())
    print("Total rows:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())