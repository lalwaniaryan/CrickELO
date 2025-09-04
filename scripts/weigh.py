# exact_like_sheet.py
# Input : "odi data.csv"  (Player, Runs, Innings, Average, Strike Rate)
# Output: "odi weighted.csv"
# Matches Excel-style: base stats rounded (for calc), STDEV.S, Z rounded to 9 dp,
# Overall_Score from rounded Zs, original base columns preserved.

import pandas as pd
import numpy as np

INPUT_FILE  = "odi data.csv"
OUTPUT_FILE = "odi weighted(4).csv"

Z_DECIMALS = 9        # as seen in your screenshot (e.g., 6.342415966)
OVERALL_DECIMALS = 9  # to match displayed sheet

def normalize_cols(df):
    df = df.copy()
    df.columns = (df.columns.str.strip()
                            .str.lower()
                            .str.replace(r"\s+"," ", regex=True))
    return df

def pick_col(df, options):
    for o in options:
        if o in df.columns:
            return o
    raise KeyError(f"Missing column. Wanted one of {options}. Got {list(df.columns)}")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def z_score_sample(series):
    s = to_num(series)
    mu = s.mean()
    sd = s.std(ddof=1)  # STDEV.S
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def main():
    df_in = pd.read_csv(INPUT_FILE)
    df = normalize_cols(df_in)

    player = pick_col(df, ["player","player name","name"])
    runs   = pick_col(df, ["runs","total runs"])
    inns   = pick_col(df, ["innings","inns"])
    avg    = pick_col(df, ["avg","average","batting avg","bat avg"])
    sr     = pick_col(df, ["strike rate","sr","s/r"])

    # ----- Create a calc copy with Excel-like display rounding for inputs -----
    calc = df.copy()
    # Innings & Runs as integers for calc
    calc[inns] = to_num(calc[inns]).round(0)
    calc[runs] = to_num(calc[runs]).round(0)
    # Average & Strike Rate to 2 decimals for calc (what Excel shows)
    calc[avg]  = to_num(calc[avg]).round(2)
    calc[sr]   = to_num(calc[sr]).round(2)

    # ----- Z-scores on the calc copy (sample std) -----
    z_runs = z_score_sample(calc[runs]).round(Z_DECIMALS)
    z_avg  = z_score_sample(calc[avg]).round(Z_DECIMALS)
    z_sr   = z_score_sample(calc[sr]).round(Z_DECIMALS)

    # Overall from rounded Zs
    overall = ((z_runs + z_avg + z_sr) / 3).round(OVERALL_DECIMALS)

    # ----- Assemble output: keep original base columns exactly as in input -----
    out = df_in.copy()  # preserves original formatting/values
    out["Z_Runs"]        = z_runs
    out["Z_Average"]     = z_avg
    out["Z_Strike Rate"] = z_sr
    out["Overall_Score"] = overall

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()