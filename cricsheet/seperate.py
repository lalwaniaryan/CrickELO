import pandas as pd
import numpy as np

# --- paths ---
FULL_DATA         = "player_match_by_match_stats_sorted(5).csv"
NUMERIC_FILE      = "numeric_with_player(5).csv"
CATEGORICAL_FILE  = "categorical_with_player(8).csv"

# --- load full dataset ---
df = pd.read_csv(FULL_DATA)

# --- detect numeric vs categorical ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Always keep these in numeric (if present)
force_numeric = [c for c in ["Batting_Position", "Position_Mode"] if c in df.columns]

# --- build numeric dataset ---
keep_numeric = ["Match_ID", "Player"]
keep_numeric += [c for c in numeric_cols if c not in keep_numeric]
# add forced cols if they weren't detected numeric
keep_numeric += [c for c in force_numeric if c not in keep_numeric]

numeric_df = df[keep_numeric].copy()

# --- build categorical dataset (NO Match_ID) ---
keep_categorical = ["Player"] + [c for c in categorical_cols if c != "Match_ID"]
categorical_df = df[keep_categorical].copy()

# --- helper: cast whole-number floats to Int64 (no .0 in CSV) ---
def cast_whole_floats_to_int(df_in, exclude=()):
    df_out = df_in.copy()
    for col in df_out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_float_dtype(df_out[col]):
            s = df_out[col].dropna()
            # if all non-null values are whole numbers, cast to Int64
            if not s.empty and np.all(np.isclose(s % 1, 0)):
                df_out[col] = df_out[col].astype("Int64")
    return df_out

# --- force position columns to Int64 if present (no decimals) ---
for col in force_numeric:
    if col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").astype("Int64")

# --- only allow decimals where necessary (e.g., Team_Won can be 0.5) ---
# Exclude Team_Won from integer casting so 0.5 remains; all other whole-number floats become Int64
exclude_from_int_cast = tuple(c for c in ["Team_Won"] if c in numeric_df.columns)
numeric_df = cast_whole_floats_to_int(numeric_df, exclude=exclude_from_int_cast)

# --- save outputs ---
numeric_df.to_csv(NUMERIC_FILE, index=False)
categorical_df.to_csv(CATEGORICAL_FILE, index=False)

print(f"✅ Saved numeric file: {NUMERIC_FILE} ({len(numeric_df)} rows × {len(numeric_df.columns)} cols)")
print(f"✅ Saved categorical file: {CATEGORICAL_FILE} ({len(categorical_df)} rows × {len(categorical_df.columns)} cols)")
