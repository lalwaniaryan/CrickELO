# elo_weighted_rank.py
# Input : player_match_by_match_stats.csv (default) OR env override
# Output: player_elo_ratings.csv, match_scores.csv
#
# Wires added:
# - Reads INPUT_FILE_OVERRIDE, PERF_SCALE_OVERRIDE, K_BASE_OVERRIDE, Z_CLIP_OVERRIDE from env
# - NEW: Reads ELO_SCALE_OVERRIDE, PERF_STRETCH_OVERRIDE
# - ELO_START fixed at 1500 as requested

import os
import pandas as pd
import numpy as np

# ------------------------------
# Defaults (can be overridden by env)
# ------------------------------
INPUT_FILE = os.getenv("INPUT_FILE_OVERRIDE", "player_match_by_match_stats.csv")

PLAYER_COL_FALLBACKS = ["player", "player_name", "name"]
DATE_COL_FALLBACKS   = ["date", "match_date"]
MATCH_ID_FALLBACKS   = ["match_id", "id"]

WEIGHTS = {
    "player": 0, "match_id": 0, "date": 0,
    "opponent": 7, "event_name": 0, "event_stage": 8, "match_type_number": 3, "venue": 4,
    "player_of_match": 4, "toss_win": 0.5, "toss_decision": 0.5, "team_won": 2,
    "win_type": 2, "win_margin": 2, "runs_scored": 28, "balls_faced": 10, "dismissal_kind": 5,
    "dismissed_by": 0, "caught_by": 0, "fours": 8, "sixes": 10, "wides_faced": 0.5, "noballs_faced": 0.5,
    "batting_position": 6, "balls_bowled": 0, "runs_conceded": 0, "wickets": 0, "wicket_kinds": 0,
    "wides_bowled": 0, "noballs_bowled": 0, "position_mode": 0,
}

CATEGORICAL = {
    "opponent","event_name","event_stage","venue","dismissal_kind","dismissed_by","caught_by",
    "toss_decision","win_type","position_mode","player_of_match"
}
BOOLEANISH = {"toss_win", "team_won"}
EXTRA_NUMERIC = {
    "match_type_number","win_margin","runs_scored","balls_faced","fours","sixes","wides_faced",
    "noballs_faced","batting_position","balls_bowled","runs_conceded","wickets","wicket_kinds",
    "wides_bowled","noballs_bowled"
}

# ------------------------------
# Elo params
# ------------------------------
ELO_START = 2000.0  # FIXED baseline
ELO_SCALE = float(os.getenv("ELO_SCALE_OVERRIDE", 900.0))  # was 400; larger => wider rating spread

K_BASE = float(os.getenv("K_BASE_OVERRIDE", 20.0))
K_MIN, K_MAX = 20.0, 200.0

PERF_SCALE = float(os.getenv("PERF_SCALE_OVERRIDE", 0.75))
Z_CLIP = float(os.getenv("Z_CLIP_OVERRIDE", 4.0))

# NEW: separation factor for WeightedPerf before logistic (larger => scores closer to 0/1)
PERF_STRETCH = float(os.getenv("PERF_STRETCH_OVERRIDE", 3.0))

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (out.columns.str.strip().str.lower().str.replace(r"\s+","_", regex=True))
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
    feats = {}
    for col, w in WEIGHTS.items():
        if col not in df.columns or w == 0:
            continue
        s = df[col]
        if col in BOOLEANISH:
            s_num = (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype(float)
        elif col in CATEGORICAL:
            s_num = label_encode(s)
        else:
            s_num = to_num(s).fillna(0.0)
        feats[col] = zscore_sample(s_num)
    if not feats:
        raise RuntimeError("No weighted features found with weight > 0.")
    return pd.DataFrame(feats, index=df.index)

def compute_weighted_perf(X: pd.DataFrame) -> pd.Series:
    used = {c: WEIGHTS[c] for c in X.columns if WEIGHTS.get(c, 0) != 0}
    w = np.array([used[c] for c in X.columns], dtype=float)
    denom = np.sum(np.abs(w))
    if denom == 0:
        raise RuntimeError("Sum of absolute weights is zero.")
    return pd.Series(X.values @ w / denom, index=X.index)

def logistic_score(perf: pd.Series, scale: float = PERF_SCALE) -> pd.Series:
    # Critical change: increase separation before logistic
    perf2 = perf * PERF_STRETCH
    return 1.0 / (1.0 + np.exp(-perf2 / scale))

def elo_expectation(r_player: float, r_baseline: float = ELO_START, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10 ** ((r_baseline - r_player) / scale))

def importance_scale(row: pd.Series) -> float:
    """
    Small K multiplier based on match context. Kept conservative.
    Note: this uses per-row encoding (not global), so it's a mild heuristic only.
    """
    s = 0.0
    def add(val, w): return w * val if pd.notna(val) else 0.0

    stage_w = WEIGHTS.get("event_stage", 0)
    win_type_w = WEIGHTS.get("win_type", 0)
    win_margin_w = WEIGHTS.get("win_margin", 0)
    venue_w = WEIGHTS.get("venue", 0)

    stage_num = label_encode(pd.Series([row.get("event_stage", "NA")]))[0] if "event_stage" in row else 0.0
    venue_num = label_encode(pd.Series([row.get("venue", "NA")]))[0] if "venue" in row else 0.0
    win_type_num = label_encode(pd.Series([row.get("win_type", "NA")]))[0] if "win_type" in row else 0.0
    win_margin_num = pd.to_numeric(pd.Series([row.get("win_margin", 0)]), errors="coerce").fillna(0.0)[0]

    stage_num = np.tanh(stage_num/5.0)
    venue_num = np.tanh(venue_num/20.0)
    win_type_num = np.tanh(win_type_num/5.0)
    win_margin_num = np.tanh(win_margin_num/50.0)

    s += add(stage_num, stage_w)
    s += add(venue_num, venue_w * 0.5)
    s += add(win_type_num, win_type_w * 0.5)
    s += add(win_margin_num, win_margin_w)

    # Keep multiplier bounded (~[1.0, 1.5]) centered around ~1.25
    mult = 1.0 + 0.25 * np.tanh(s / 20.0) + 0.25
    return float(mult)

def main():
    df_raw = pd.read_csv(INPUT_FILE)
    df = normalize_cols(df_raw)

    player_col = pick_col(df, PLAYER_COL_FALLBACKS)
    date_col   = pick_col(df, DATE_COL_FALLBACKS)
    match_col  = pick_col(df, MATCH_ID_FALLBACKS)

    # ensure all weighted cols exist
    for key in WEIGHTS.keys():
        if key not in df.columns:
            df[key] = np.nan

    X = build_feature_matrix(df)
    weighted_perf = compute_weighted_perf(X)
    observed = logistic_score(weighted_perf)

    # stable ordering
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_order"] = np.lexsort((
        df[match_col].astype(str).values,
        df["_parsed_date"].fillna(pd.Timestamp("1970-01-01")).values
    ))
    order_idx = np.argsort(df["_order"].values)

    elo, played, rows = {}, {}, []

    for idx in order_idx:
        row = df.iloc[idx]
        p = row[player_col]
        s = float(observed.iloc[idx])

        rating_before = float(elo.get(p, ELO_START))
        E = elo_expectation(rating_before, r_baseline=ELO_START, scale=ELO_SCALE)

        k_mult = importance_scale(row)
        K = float(np.clip(K_BASE * k_mult, K_MIN, K_MAX))

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
            "Elo_Expected": float(E),
            "K_used": K,
            "Elo_After": float(rating_after),
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

    pd.DataFrame(rows).to_csv("match_scores.csv", index=False)

    final = pd.DataFrame({
        "Player": list(elo.keys()),
        "Elo": list(elo.values()),
        "Matches_Played": [played[p] for p in elo.keys()]
    }).sort_values(["Elo","Matches_Played"], ascending=[False, False])

    final.to_csv("player_elo_ratings.csv", index=False)

    print(
        "Saved: player_elo_ratings.csv, match_scores.csv "
        f"(from {INPUT_FILE}) | ELO_START=1500 | ELO_SCALE={ELO_SCALE} | PERF_STRETCH={PERF_STRETCH}"
    )

if __name__ == "__main__":
    main()
