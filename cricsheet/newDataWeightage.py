# elo_weighted_rank.py
# Input : player_match_by_match_stats.csv
# Output: player_elo_ratings.csv, match_scores.csv
#
# What this does:
# 1) Standardizes weighted features, forms a per-row WeightedPerf.
# 2) Maps WeightedPerf -> observed score s in [0,1] via logistic.
# 3) Solo-Elo per player: rating_t+1 = rating_t + K * (s - E), with E from Elo expectation vs 1500 baseline.
# 4) K is scaled by your "importance" weights (Event_Stage, Win_Type, Win_Margin, Venue, etc.).
#
# Notes:
# - This is a *player-vs-field* Elo (no opponent Elo needed).
# - Categorical columns are label-encoded then z-scored (simple, deterministic).
#   For higher fidelity, replace with target-encoding later (see recommendations below).

import pandas as pd
import numpy as np

INPUT_FILE = "player_match_by_match_stats.csv"
PLAYER_COL_FALLBACKS = ["player", "player_name", "name"]
DATE_COL_FALLBACKS   = ["date", "match_date"]
MATCH_ID_FALLBACKS   = ["match_id", "id"]

# ---- Your ODI Weights ----
WEIGHTS = {
    "player": 0,
    "match_id": 0,
    "date": 0,
    "opponent": 7,
    "event_name": 0,
    "event_stage": 8,
    "match_type_number": 3,
    "venue": 4,
    "player_of_match": 4,
    "toss_win": 0.5,
    "toss_decision": 0.5,
    "team_won": 2,
    "win_type": 2,
    "win_margin": 2,
    "runs_scored": 28,
    "balls_faced": 10,
    "dismissal_kind": 5,
    "dismissed_by": 0,
    "caught_by": 0,
    "fours": 8,
    "sixes": 10,
    "wides_faced": 0.5,
    "noballs_faced": 0.5,
    "batting_position": 6,
    "balls_bowled": 0,
    "runs_conceded": 0,
    "wickets": 0,
    "wicket_kinds": 0,
    "wides_bowled": 0,
    "noballs_bowled": 0,
    "position_mode": 0,
}

# Which features are categorical (simple label encoding for v1)
CATEGORICAL = {
    "opponent", "event_name", "event_stage", "venue",
    "dismissal_kind", "dismissed_by", "caught_by",
    "toss_decision", "win_type", "position_mode", "player_of_match"
}

# Which features are boolean-like (coerce to 0/1)
BOOLEANISH = {"toss_win", "team_won"}

# Features that are numeric (coerce to numeric)
# We'll infer numerics by: weight > 0 and not in CATEGORICAL or BOOLEANISH
# But we also include common numeric columns explicitly:
EXTRA_NUMERIC = {
    "match_type_number", "win_margin",
    "runs_scored", "balls_faced", "fours", "sixes",
    "wides_faced", "noballs_faced", "batting_position",
    "balls_bowled", "runs_conceded", "wickets",
    "wicket_kinds", "wides_bowled", "noballs_bowled"
}

# Elo params
ELO_START = 1500.0
ELO_SCALE = 400.0

# Base K; will be scaled by match importance
K_BASE = 20.0
# Cap K to be sane
K_MIN, K_MAX = 8.0, 60.0

# For logistic mapping of performance -> observed score in [0,1]
# s = 1 / (1 + exp(-perf / PERF_SCALE))
PERF_SCALE = 0.75  # lower -> more "peaky". Tune later.

# Z-score protection
Z_CLIP = 4.0  # clip z-scores to [-4, 4] to avoid outlier explosions

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (out.columns
                   .str.strip()
                   .str.lower()
                   .str.replace(r"\s+", "_", regex=True))
    return out

def pick_col(df, options):
    for o in options:
        if o in df.columns:
            return o
    raise KeyError(f"Missing one of columns: {options}. Found: {list(df.columns)}")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore_sample(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    z = (x - mu) / sd
    return z.clip(-Z_CLIP, Z_CLIP)

def label_encode(series: pd.Series) -> pd.Series:
    vals = series.astype(str).fillna("NA")
    codes, _ = pd.factorize(vals, sort=True)
    return pd.Series(codes, index=series.index, dtype="float64")

def build_feature_matrix(df: pd.DataFrame):
    # Prepare each feature to numeric, then z-score if weight>0
    feats = {}
    for col, w in WEIGHTS.items():
        if col not in df.columns or w == 0:
            continue

        s = df[col]
        if col in BOOLEANISH:
            # Map truthy to 1.0, else 0.0
            s_num = s.fillna(0)
            if s_num.dtype == bool:
                s_num = s_num.astype(float)
            else:
                # Many sources encode as 0/1 already; any nonzero -> 1
                s_num = (pd.to_numeric(s_num, errors="coerce").fillna(0) != 0).astype(float)
        elif col in CATEGORICAL:
            s_num = label_encode(s)
        elif col in EXTRA_NUMERIC or (w > 0 and col not in CATEGORICAL):
            s_num = to_num(s).fillna(0.0)
        else:
            s_num = to_num(s).fillna(0.0)

        feats[col] = zscore_sample(s_num)  # standardized for comparability

    if not feats:
        raise RuntimeError("No weighted features found in the dataset with weight > 0.")

    X = pd.DataFrame(feats, index=df.index)
    return X

def compute_weighted_perf(X: pd.DataFrame) -> pd.Series:
    # Weighted sum of z-scores normalized by sum of absolute weights in X
    used_weights = {c: WEIGHTS[c] for c in X.columns if WEIGHTS.get(c, 0) != 0}
    w = np.array([used_weights[c] for c in X.columns], dtype=float)
    denom = np.sum(np.abs(w))
    if denom == 0:
        raise RuntimeError("Sum of absolute weights is zero.")
    perf = (X.values @ w) / denom
    return pd.Series(perf, index=X.index)

def logistic_score(perf: pd.Series, scale: float = PERF_SCALE) -> pd.Series:
    # Map real-valued perf to [0,1] smoothly
    return 1.0 / (1.0 + np.exp(-perf / scale))

def elo_expectation(r_player: float, r_baseline: float = 1500.0, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10 ** ((r_baseline - r_player) / scale))

def importance_scale(row: pd.Series) -> float:
    """
    Scale K by match context using your weights and current row's standardized values.
    We recompute a small 'context score' from a subset of context features.
    Using raw (already z-scored in X) would be best, but we only have the row here.
    So we approximate by reusing the same z-style transforms prepared earlier.
    For simplicity, we build a small linear combo based on present numeric-ish values.
    """
    # Pull raw columns if exist; coerce to numeric/label-encode on the fly, z-score with dataset stats would be better.
    # To keep it simple and deterministic, we rely on *precomputed* z columns if we appended them, else fall back.
    # Here we’ll derive a soft scaler from a few readily interpretable fields:

    # Event importance (stage, type), venue, win context
    s = 0.0
    add = lambda val, w: w * val if pd.notna(val) else 0.0

    # Mild signals: give small boosts for later stages / decisive wins / known venues
    stage_w = WEIGHTS.get("event_stage", 0)
    win_type_w = WEIGHTS.get("win_type", 0)
    win_margin_w = WEIGHTS.get("win_margin", 0)
    venue_w = WEIGHTS.get("venue", 0)

    # Simple numeric proxies:
    stage_num = label_encode(pd.Series([row.get("event_stage", "NA")]))[0] if "event_stage" in row else 0.0
    venue_num = label_encode(pd.Series([row.get("venue", "NA")]))[0] if "venue" in row else 0.0
    win_type_num = label_encode(pd.Series([row.get("win_type", "NA")]))[0] if "win_type" in row else 0.0
    win_margin_num = pd.to_numeric(pd.Series([row.get("win_margin", 0)]), errors="coerce").fillna(0.0)[0]

    # Normalize rough scales to ~z-like ranges
    stage_num = np.tanh(stage_num / 5.0)
    venue_num = np.tanh(venue_num / 20.0)
    win_type_num = np.tanh(win_type_num / 5.0)
    win_margin_num = np.tanh(win_margin_num / 50.0)

    s += add(stage_num, stage_w)
    s += add(venue_num, venue_w * 0.5)
    s += add(win_type_num, win_type_w * 0.5)
    s += add(win_margin_num, win_margin_w)

    # Convert to K multiplier ~ [0.8, 1.5] after a smooth squash
    mult = 1.0 + 0.25 * np.tanh(s / 20.0) + 0.25  # 1.25 mid, +/- ~0.25
    return float(mult)

def main():
    df_raw = pd.read_csv(INPUT_FILE)
    df = normalize_cols(df_raw)

    player_col = pick_col(df, PLAYER_COL_FALLBACKS)
    date_col   = pick_col(df, DATE_COL_FALLBACKS)
    match_col  = pick_col(df, MATCH_ID_FALLBACKS)

    # Ensure all potential weight columns exist (add empty if missing)
    for key in WEIGHTS.keys():
        if key not in df.columns:
            df[key] = np.nan

    # Build standardized feature matrix for weighted perf
    X = build_feature_matrix(df)
    weighted_perf = compute_weighted_perf(X)
    observed = logistic_score(weighted_perf)

    # Sort by time (if date exists)
    # If date has weird formatting, pandas will coerce best-effort
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_order"] = np.lexsort((
        df[match_col].astype(str).values,
        df["_parsed_date"].fillna(pd.Timestamp("1970-01-01")).values
    ))
    order_idx = np.argsort(df["_order"].values)

    # Elo state
    elo = {}
    played = {}

    rows = []
    for idx in order_idx:
        row = df.iloc[idx]
        p = row[player_col]
        s = float(observed.iloc[idx])

        rating_before = elo.get(p, ELO_START)
        E = elo_expectation(rating_before, r_baseline=ELO_START, scale=ELO_SCALE)

        k_mult = importance_scale(row)
        K = np.clip(K_BASE * k_mult, K_MIN, K_MAX)

        rating_after = rating_before + K * (s - E)

        elo[p] = float(rating_after)
        played[p] = played.get(p, 0) + 1

        rows.append({
            "Player": p,
            "Match_ID": row[match_col],
            "Date": row[date_col],
            "WeightedPerf": float(weighted_perf.iloc[idx]),
            "ObservedScore": s,
            "Elo_Before": rating_before,
            "Elo_Expected": E,
            "K_used": float(K),
            "Elo_After": rating_after,
            "Opponent": row.get("opponent", np.nan),
            "Venue": row.get("venue", np.nan),
            "Event_Stage": row.get("event_stage", np.nan),
            "Win_Type": row.get("win_type", np.nan),
            "Win_Margin": row.get("win_margin", np.nan),
            "Runs_Scored": row.get("runs_scored", np.nan),
            "Balls_Faced": row.get("balls_faced", np.nan),
            "Fours": row.get("fours", np.nan),
            "Sixes": row.get("sixes", np.nan),
            "Team_Won": row.get("team_won", np.nan),
        })

    match_out = pd.DataFrame(rows)
    match_out.to_csv("match_scores.csv", index=False)

    final = pd.DataFrame({
        "Player": list(elo.keys()),
        "Elo": list(elo.values()),
        "Matches_Played": [played[p] for p in elo.keys()]
    }).sort_values(["Elo", "Matches_Played"], ascending=[False, False])

    final.to_csv("player_elo_ratings.csv", index=False)
    print("Saved: player_elo_ratings.csv, match_scores.csv")

if __name__ == "__main__":
    main()
