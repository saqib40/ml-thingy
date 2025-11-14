import os
import numpy as np
import pandas as pd

def clean_file(inpath, outdir="../data"):
    df = pd.read_csv(inpath)

    specialColumnsM = [
        "PoolQC", "Fence", "Alley", "MiscFeature",
        "FireplaceQu", "GarageType", "GarageFinish",
        "GarageQual", "GarageCond"
    ]

    threshold = len(df) / 2
    for col in list(df.columns):
        if (df[col].isna().sum() > threshold) and (col not in specialColumnsM):
            df.drop(columns=col, inplace=True)

    for col in specialColumnsM:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    numericColumns = df.select_dtypes(include=[np.number]).columns
    for col in numericColumns:
        if df[col].isna().sum() > 0:
            if abs(df[col].skew()) < 1:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())

    categoricalColumns = df.select_dtypes(include=["object"]).columns
    for col in categoricalColumns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    ord_mappings = {
        "ExterQual":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "ExterCond":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "BsmtQual":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "BsmtCond":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "HeatingQC":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "GarageQual":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "GarageCond":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
        "PoolQC":      {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0},
        "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
        "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
        "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
        "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
        "PavedDrive":   {"Y": 2, "P": 1, "N": 0, "None": 0},
        "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0, "None": 0}
    }

    for col, mapping in ord_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    
    remaining_obj_cols = df.select_dtypes(include="object").columns.tolist()

    if "Id" in remaining_obj_cols:
        remaining_obj_cols.remove("Id")

    if remaining_obj_cols:
        df = pd.get_dummies(df, columns=remaining_obj_cols, drop_first=True)

    # write output file
    base = os.path.basename(inpath)
    outname = f"clean_{base}"
    outpath = os.path.join(outdir, outname)
    df.to_csv(outpath, index=False)
    print("Wrote:", outpath)
    return df

if __name__ == "__main__":
    # example usage:
    # python cleanData.py ../data/train.csv
    import sys
    infile = sys.argv[1] if len(sys.argv) > 1 else "../data/train.csv"
    clean_file(infile, outdir="../data")
