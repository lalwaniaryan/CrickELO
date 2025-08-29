#Sample code
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ODI player match-by-match stats from Cricsheet-style JSON, then compute per-player
aggregates and weighted Z-scores (Runs, Average, Strike Rate) with a composite Overall_Score.

Key outputs:
- player_match_by_match_stats_sorted.csv  (long/row-wise per player per match)
- player_aggregate_with_zscores.csv       (one row per player with Z metrics)

You can extend this with opponent-aware Elo later; a stub is provided at the bottom.
"""

import os
import json
import math
import warnings
from typing import Dict, List, Any, Tuple

import pandas as pd
from tqdm import tqdm


# ==============================
# 1) CONFIGURATION
# ==============================
PLAYER_LIST_PATH = "ODI weighted.csv"   # CSV or Excel with a "Player" column
PLAYER_COLUMN_NAME = "Player"

DATA_FOLDER = "odis_json"               # Folder containing Cricsheet-style ODI JSON files
MATCH_BY_MATCH_CSV = "player_match_by_match_stats_sorted.csv"
AGG_ZSCORES_CSV = "player_aggregate_with_zscores.csv"

# Weights for overall score (tweak to taste)
WEIGHT_Z_RUNS = 0.4
WEIGHT_Z_AVG = 0.35
WEIGHT_Z_SR  = 0.25


# ==============================
# 2) UTILS
# ==============================
def _safe_lower_strip(x: Any) -> str:
    """Lowercase + strip a string safely; return empty string for None."""
    if isinstance(x, str):
        return x.lower().strip()
    return ""


def _titlecase_name(x: str) -> str:
    """Simple .title() for player names; replace as needed if you have your own mapping."""
    return x.title()


def _calc_strike_rate(runs: int, balls: int) -> float:
    """Strike Rate = 100 * runs / balls. Return NaN if balls == 0."""
    if balls and balls > 0:
        return 100.0 * runs / balls
    return float("nan")


def _zscore(series: pd.Series) -> pd.Series:
    """Standard z-score with protection for zero std."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or math.isnan(sigma):
        # All equal or empty -> return zeros
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mu) / sigma


# ==============================
# 3) LOAD TARGET PLAYERS
# ==============================
def load_target_players(path: str, col: str) -> pd.DataFrame:
    """
    Load the player list from CSV or Excel; returns DataFrame with at least 'Player' col.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Player list file not found: {path}")

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}. Columns: {df.columns.tolist()}")

    return df


# ==============================
# 4) PARSE ONE MATCH FILE
# ==============================
def parse_match_file(filepath: str, target_players_lc: set) -> List[Dict[str, Any]]:
    """
    Parse a single Cricsheet-style JSON file and return a list of per-player match rows.

    Notes / Fixes vs your original:
    - We compute opponent **per player** based on the innings where they appeared,
      so players on both teams in a match are correctly associated.
    - We carry the venue into the per-player records.
    - We keep dismissal_kind (default 'Not Out') and set is_dismissed=1 for player_out.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    info = match_data.get("info", {})
    if info.get("match_type") != "ODI":
        return []

    match_id = os.path.basename(filepath).replace(".json", "")
    match_date = info.get("dates", [None])[0]  # Cricsheet stores a list of dates
    teams = info.get("teams", [])
    venue = info.get("venue", "N/A")

    event_info = info.get("event", {}) or {}
    event_name = event_info.get("name", "Bilateral Series")
    stage = event_info.get("stage", "N/A")

    pom_list = [ _safe_lower_strip(x) for x in (info.get("player_of_match", []) or []) ]

    # Stats keyed by player_lower: we also store the opponent they faced for their appearances.
    # If a player appears in multiple innings (rare in ODIs), we keep first non-empty opponent seen.
    per_player: Dict[str, Dict[str, Any]] = {}

    for inning in match_data.get("innings", []):
        team_batting = inning.get("team")
        if not teams or len(teams) != 2:
            # Defensive: if teams missing/malformed, skip opponent assignment
            batting_team = team_batting
            opponent_team = "Unknown"
        else:
            # Opponent is the other team
            opponent_team = teams[1] if teams[0] == team_batting else teams[0]

        for over in inning.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = _safe_lower_strip(delivery.get("batter"))
                bowler = _safe_lower_strip(delivery.get("bowler"))

                # Initialize entries for any target player we see (as batter or bowler)
                for p in (batter, bowler):
                    if p in target_players_lc and p not in per_player:
                        per_player[p] = {
                            "runs_scored": 0, "balls_faced": 0, "fours": 0, "sixes": 0,
                            "balls_bowled": 0, "runs_conceded": 0, "wickets_taken": 0,
                            "is_dismissed": 0, "dismissal_kind": "Not Out",
                            "opponent": None  # filled when we know which side they faced
                        }

                # Batting stats
                if batter in target_players_lc:
                    runs_obj = delivery.get("runs", {}) or {}
                    batter_runs = int(runs_obj.get("batter", 0) or 0)
                    per_player[batter]["runs_scored"] += batter_runs
                    per_player[batter]["balls_faced"] += 1
                    if batter_runs == 4:
                        per_player[batter]["fours"] += 1
                    if batter_runs == 6:
                        per_player[batter]["sixes"] += 1
                    # Opponent for a batter is the bowling team (i.e., the other team)
                    if per_player[batter]["opponent"] is None:
                        per_player[batter]["opponent"] = opponent_team

                # Bowling stats
                if bowler in target_players_lc:
                    runs_total = int((delivery.get("runs", {}) or {}).get("total", 0) or 0)
                    per_player[bowler]["balls_bowled"] += 1
                    per_player[bowler]["runs_conceded"] += runs_total

                    # Credit wickets except those not attributed to bowler
                    wicket = delivery.get("wicket")
                    if wicket:
                        kind = (wicket.get("kind") or "").lower()
                        if kind not in ["run out", "retired hurt", "obstructing the field"]:
                            per_player[bowler]["wickets_taken"] += 1

                    # Opponent for a bowler is the batting team (i.e., team_batting)
                    if per_player[bowler]["opponent"] is None:
                        per_player[bowler]["opponent"] = team_batting

                # Dismissal info (for batter out)
                wicket = delivery.get("wicket")
                if wicket:
                    player_out = _safe_lower_strip(wicket.get("player_out"))
                    if player_out in target_players_lc:
                        per_player[player_out]["is_dismissed"] = 1
                        per_player[player_out]["dismissal_kind"] = wicket.get("kind", "N/A")

    # Build rows (one per player in this match)
    rows: List[Dict[str, Any]] = []
    for p_lc, stats in per_player.items():
        rows.append({
            "Player": _titlecase_name(p_lc),
            "Match_ID": match_id,
            "Date": match_date,
            "Opponent": stats.get("opponent") or "Unknown",
            "Event_Name": event_name,
            "Event_Stage": stage,
            "Player_of_Match": 1 if p_lc in pom_list else 0,
            "Runs_Scored": stats["runs_scored"],
            "Balls_Faced": stats["balls_faced"],
            "Dismissal_Kind": stats["dismissal_kind"],
            "Fours": stats["fours"],
            "Sixes": stats["sixes"],
            "Is_Dismissed": stats["is_dismissed"],
            "Balls_Bowled": stats["balls_bowled"],
            "Runs_Conceded": stats["runs_conceded"],
            "Wickets_Taken": stats["wickets_taken"],
            "Venue": venue,
        })

    return rows


# ==============================
# 5) PROCESS FOLDER
# ==============================
def build_match_by_match(
    data_folder: str,
    target_players_df: pd.DataFrame,
    player_col: str
) -> pd.DataFrame:
    """
    Iterate all JSON files and build the long match-by-match DataFrame for target players.
    """
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder not found: {data_folder}")

    target_players_lc = set(
        target_players_df[player_col].astype(str).str.lower().str.strip().tolist()
    )

    json_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".json")])
    print(f"Processing {len(json_files)} match files...")

    all_rows: List[Dict[str, Any]] = []
    for fname in tqdm(json_files, desc="Parsing Matches"):
        fpath = os.path.join(data_folder, fname)
        try:
            rows = parse_match_file(fpath, target_players_lc)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Skipping {fname} due to error: {e}")

    if not all_rows:
        warnings.warn("No rows parsed. Check your JSON folder and filters.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Sort by Player then Date (ensure dates sort correctly)
    # Cricsheet dates are ISO-like; to be safe, coerce to datetime.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.sort_values(by=["Player", "Date", "Match_ID"], inplace=True, kind="mergesort")

    # Reorder columns for readability
    cols = [
        "Player", "Match_ID", "Date", "Opponent", "Event_Name", "Event_Stage",
        "Player_of_Match", "Runs_Scored", "Balls_Faced", "Dismissal_Kind", "Fours",
        "Sixes", "Is_Dismissed", "Balls_Bowled", "Runs_Conceded", "Wickets_Taken", "Venue"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df


# ==============================
# 6) AGGREGATE + Z-SCORES
# ==============================
def aggregate_player_level(mbm: pd.DataFrame) -> pd.DataFrame:
    """
    Create a per-player aggregate table with:
      - Innings (batting innings observed)
      - Runs (sum)
      - Average (runs / dismissals)  [dismissal counted when Is_Dismissed==1]
      - Strike Rate (100 * sum(runs) / sum(balls_faced))
      - Z-scores for Runs, Average, Strike Rate
      - Overall_Score (weighted sum of Zs)
    """
    if mbm.empty:
        return pd.DataFrame()

    # Batting innings counted as rows where Balls_Faced > 0 (ODI-wide convention for this dataset)
    batting = mbm.copy()
    batting["BF_valid"] = batting["Balls_Faced"].fillna(0).astype(int)
    batting["Runs_valid"] = batting["Runs_Scored"].fillna(0).astype(int)
    batting["Out_flag"] = batting["Is_Dismissed"].fillna(0).astype(int)

    grouped = batting.groupby("Player", as_index=False).agg(
        Innings=("BF_valid", lambda x: int((x > 0).sum())),
        Runs=("Runs_valid", "sum"),
        Balls=("BF_valid", "sum"),
        Outs=("Out_flag", "sum"),
    )

    # Averages and SR
    grouped["Average"] = grouped.apply(
        lambda r: (r["Runs"] / r["Outs"]) if r["Outs"] > 0 else float("nan"),
        axis=1
    )
    grouped["Strike Rate"] = grouped.apply(
        lambda r: _calc_strike_rate(int(r["Runs"]), int(r["Balls"])),
        axis=1
    )

    # Z-scores
    grouped["Z_Runs"] = _zscore(grouped["Runs"])
    grouped["Z_Average"] = _zscore(grouped["Average"])
    grouped["Z_Strike Rate"] = _zscore(grouped["Strike Rate"])

    # Composite (weights configurable up top)
    grouped["Overall_Score"] = (
        WEIGHT_Z_RUNS * grouped["Z_Runs"]
        + WEIGHT_Z_AVG * grouped["Z_Average"]
        + WEIGHT_Z_SR  * grouped["Z_Strike Rate"]
    )

    # Order roughly like your example
    grouped = grouped[
        ["Player", "Innings", "Runs", "Average", "Strike Rate",
         "Z_Runs", "Z_Average", "Z_Strike Rate", "Overall_Score"]
    ].sort_values("Overall_Score", ascending=False)

    return grouped


# ==============================
# 7) (OPTIONAL) ELO STUB
# ==============================
def update_elo_stub():
    """
    Placeholder for opponent-aware Elo.
    Idea:
      - Maintain dictionaries: player_rating[player], team_rating[team]
      - For each match row:
          * define performance outcome per player vs opponent team (e.g., compare player's
            batting Z in that match to opponent team's bowling Z for that match)
          * expected = 1 / (1 + 10^(-(R_player - R_team)/400))
          * rating_new = rating_old + K * (outcome - expected)
      - Persist ratings by chronological Date order.

    You already have per-match rows (mbm DataFrame). To build per-match Zs,
    group within a match_id and compute player-relative stats, then compare to
    opponent aggregates.

    Left as a stub because the exact outcome definition is your design choice.
    """
    pass


# ==============================
# 8) MAIN
# ==============================
def main():
    # 1) Load players
    players_df = load_target_players(PLAYER_LIST_PATH, PLAYER_COLUMN_NAME)
    print(f"Loaded {len(players_df)} players from '{PLAYER_LIST_PATH}'.")

    # 2) Build match-by-match long table
    mbm = build_match_by_match(DATA_FOLDER, players_df, PLAYER_COLUMN_NAME)
    if mbm.empty:
        print("No match-by-match rows generated. Exiting.")
        return

    # 3) Save match-by-match CSV
    mbm.to_csv(MATCH_BY_MATCH_CSV, index=False)
    print(f"✅ Saved match-by-match table: {MATCH_BY_MATCH_CSV}  (rows: {len(mbm)})")

    # 4) Aggregate + Z-scores
    agg = aggregate_player_level(mbm)
    if not agg.empty:
        agg.to_csv(AGG_ZSCORES_CSV, index=False)
        print(f"✅ Saved aggregate + z-scores: {AGG_ZSCORES_CSV}  (players: {len(agg)})")

    # 5) (Optional) Elo prototype
    # update_elo_stub()


if __name__ == "__main__":
    main()
