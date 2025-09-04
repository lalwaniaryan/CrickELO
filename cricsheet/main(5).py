# MAIN(1).py — Batters + optional bowling, with rich context (+Venue, bowler wides/noballs)
import os
import json
import math
import pandas as pd
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP

# ---------- PATHS ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_LIST_PATH = os.path.join(SCRIPT_DIR, "ODI weighted.csv")        # must have column 'Player'
JSON_DIR        = os.path.join(SCRIPT_DIR, "odis_json")                # folder with .json
OUTPUT_CSV      = os.path.join(SCRIPT_DIR, "player_match_by_match_stats_sorted(8).csv")
PLAYER_COL = "Player"

# ---------- RULES ----------
def legal_ball_faced(delivery: dict) -> bool:
    extras = delivery.get("extras", {}) or {}
    return ("wides" not in extras) and ("no_balls" not in extras)

def legal_ball_bowled(delivery: dict) -> bool:
    extras = delivery.get("extras", {}) or {}
    return ("wides" not in extras) and ("no_balls" not in extras)

BOWLER_WICKET_KINDS = {"bowled", "caught", "lbw", "hit wicket", "stumped"}

# sports-style rounding (not bankers)
def r2(x):
    return None if x is None else float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# convert counted legal balls -> decimal overs (47 overs 3 balls -> 47.3)
def balls_to_decimal_overs(balls: int) -> float:
    o = balls // 6
    rem = balls % 6
    return o + rem / 10.0

print("Roster:", PLAYER_LIST_PATH)
print("JSON dir:", JSON_DIR)
print("Output:", OUTPUT_CSV)

# ---------- LOAD ROSTER ----------
if not os.path.exists(PLAYER_LIST_PATH):
    raise FileNotFoundError(f"Missing roster file: {PLAYER_LIST_PATH}")

players_df = pd.read_csv(PLAYER_LIST_PATH)
if PLAYER_COL not in players_df.columns:
    raise ValueError(f"'{PLAYER_COL}' column not found in {PLAYER_LIST_PATH}")

target_players = set(players_df[PLAYER_COL].astype(str).str.lower().str.strip())
print(f"Loaded {len(target_players)} target players.")

# ---------- PROCESS MATCHES ----------
if not os.path.isdir(JSON_DIR):
    raise FileNotFoundError(f"Missing JSON folder: {JSON_DIR}")

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} JSON matches.")

rows = []

def ensure_player(stats, name):
    if name not in stats:
        stats[name] = {
            "team": None,
            "opponent": None,
            "player_of_match": 0,

            # batting
            "runs_scored": 0,
            "balls_faced": 0,
            "fours": 0,
            "sixes": 0,
            "is_dismissed": 0,
            "dismissal_kind": "Not Out",
            "dismissed_by": "",
            "caught_by": "",
            "wides_faced": 0,
            "noballs_faced": 0,
            "bat_pos": None,

            # batting context
            "dot_balls": 0,
            "team_score_entry": None,
            "team_wkts_entry": None,
            "team_score_exit": None,
            "team_wkts_exit": None,
            "target_score": None,        # only if batting in 2nd innings
            "on_entry_req_rr": None,     # decimal-overs based, rounded 2dp
            "on_entry_curr_rr": None,    # decimal-overs based, rounded 2dp
            "on_entry_rr_diff": None,    # req - curr

            # bowling
            "balls_bowled": 0,
            "runs_conceded": 0,
            "wickets_taken": 0,
            "wicket_kinds_bowling": [],
            "wides_bowled": 0,
            "noballs_bowled": 0,
            "maidens": None,             # None until we learn he bowled

            # bowling extras/metrics
            "bowler_dot_balls": 0,
            "fours_conceded": 0,
            "sixes_conceded": 0,
        }

for fname in tqdm(json_files, desc="Processing matches"):
    fpath = os.path.join(JSON_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            match = json.load(f)

        info = match.get("info", {}) or {}
        if info.get("match_type") != "ODI":
            continue

        match_id = os.path.splitext(fname)[0]
        match_date = (info.get("dates") or [None])[0]
        event = info.get("event", {}) or {}
        event_name = event.get("name", "Bilateral Series")
        event_stage = event.get("stage", "N/A")
        match_type_number = info.get("match_type_number", None)
        venue_name = info.get("venue", None)
        teams = info.get("teams", []) or []
        toss = info.get("toss", {}) or {}
        toss_winner = toss.get("winner")
        toss_decision = toss.get("decision")
        outcome = info.get("outcome", {}) or {}
        potm_list = [n.lower().strip() for n in info.get("player_of_match", [])]

        overs_limit = int(info.get("overs", 50))
        scheduled_balls = overs_limit * 6

        innings_list = match.get("innings", []) or []
        innings_totals = []
        for inn in innings_list:
            t_runs = 0
            for ov in inn.get("overs", []) or []:
                for d in ov.get("deliveries", []) or []:
                    t_runs += int((d.get("runs", {}) or {}).get("total", 0))
            innings_totals.append({"team": inn.get("team"), "total": t_runs})
        target_second_innings = (innings_totals[0]["total"] + 1) if len(innings_totals) >= 1 else None

        stats = {}

        for inn_idx, inning in enumerate(innings_list):
            batting_team = inning.get("team")
            if not batting_team:
                continue
            opponent_team = teams[1] if len(teams) == 2 and teams[0] == batting_team else (
                teams[0] if len(teams) == 2 else next((t for t in teams if t != batting_team), None)
            )
            is_second_innings = (inn_idx == 1)

            # per-innings trackers
            batting_order = []
            first_seen_pos = {}
            inning_runs = 0
            inning_wkts = 0
            inning_legal_balls = 0

            for over in inning.get("overs", []) or []:
                # per-over maiden tracking
                over_bowler = None
                over_conceded = 0
                over_legal = 0

                for d in over.get("deliveries", []) or []:
                    runs = d.get("runs", {}) or {}
                    r_bat = int(runs.get("batter", 0))
                    r_total = int(runs.get("total", 0))
                    extras = d.get("extras", {}) or {}
                    wides = int(extras.get("wides", 0))
                    noballs = int(extras.get("no_balls", extras.get("noballs", 0)))
                    byes = int(extras.get("byes", 0))
                    legbyes = int(extras.get("legbyes", 0))
                    penalty = int(extras.get("penalty", 0))

                    legal_faced = legal_ball_faced(d)
                    legal_bowled = legal_ball_bowled(d)

                    batter = d.get("batter")
                    bowler = d.get("bowler")
                    if over_bowler is None and bowler:
                        over_bowler = bowler

                    # PRE state (before applying this ball)
                    pre_runs = inning_runs
                    pre_wkts = inning_wkts
                    pre_balls = inning_legal_balls

                    if batter and batter not in first_seen_pos:
                        batting_order.append(batter)
                        first_seen_pos[batter] = len(batting_order)

                    # ---- Batting (only when appears as BATTER) ----
                    if batter and batter.strip().lower() in target_players:
                        key = batter.strip()
                        if key not in stats:
                            ensure_player(stats, key)
                        rec_bat = stats[key]
                        if rec_bat["team"] is None:
                            rec_bat["team"] = batting_team
                            rec_bat["opponent"] = opponent_team
                            rec_bat["player_of_match"] = 1 if key.lower() in potm_list else 0

                        if rec_bat["bat_pos"] is None and batter in first_seen_pos:
                            rec_bat["bat_pos"] = first_seen_pos[batter]

                        # ON-ENTRY: freeze at decimal-overs PRE state of first ball faced
                        if rec_bat["team_score_entry"] is None:
                            rec_bat["team_score_entry"] = pre_runs
                            rec_bat["team_wkts_entry"] = pre_wkts

                            if is_second_innings:
                                rec_bat["target_score"] = target_second_innings
                                entry_overs_dec = balls_to_decimal_overs(pre_balls)  # e.g., 47.3 at 47.4 entry
                                overs_remaining_dec = max(overs_limit - entry_overs_dec, 0.0)

                                req_runs = None
                                if target_second_innings is not None:
                                    req_runs = max(target_second_innings - pre_runs, 0)

                                req_rr = (req_runs / overs_remaining_dec) if (req_runs is not None and overs_remaining_dec > 0) else None
                                curr_rr = (pre_runs / entry_overs_dec) if entry_overs_dec > 0 else 0.0
                                rr_diff = (req_rr - curr_rr) if req_rr is not None else None

                                rec_bat["on_entry_req_rr"]   = r2(req_rr)
                                rec_bat["on_entry_curr_rr"]  = r2(curr_rr)
                                rec_bat["on_entry_rr_diff"]  = r2(rr_diff)

                        # basic batting counts
                        rec_bat["runs_scored"] += r_bat
                        if legal_faced:
                            rec_bat["balls_faced"] += 1
                            if r_total == 0:
                                rec_bat["dot_balls"] += 1
                        if r_bat == 4:
                            rec_bat["fours"] += 1
                        elif r_bat == 6:
                            rec_bat["sixes"] += 1
                        rec_bat["wides_faced"] += wides
                        rec_bat["noballs_faced"] += noballs

                    # POST state
                    post_runs = pre_runs + r_total
                    post_wkts = pre_wkts + len(d.get("wickets", []) or [])
                    post_balls = pre_balls + (1 if legal_bowled else 0)

                    # ---- Dismissals (exit state for dismissed batter) ----
                    for w in d.get("wickets", []) or []:
                        out_name = (w.get("player_out") or "").strip()
                        kind = (w.get("kind") or "Not Out")
                        fielders = w.get("fielders", []) or []
                        caught_names = [fx.get("name") for fx in fielders if fx.get("name")]
                        if out_name and out_name.lower() in target_players:
                            ensure_player(stats, out_name)
                            rec_out = stats[out_name]
                            if rec_out["team"] is None:
                                rec_out["team"] = batting_team
                                rec_out["opponent"] = opponent_team
                                rec_out["player_of_match"] = 1 if out_name.lower() in potm_list else 0
                            rec_out["is_dismissed"] = 1
                            rec_out["dismissal_kind"] = kind
                            bowler_name = (d.get("bowler") or "").strip()
                            if kind.lower() in BOWLER_WICKET_KINDS and bowler_name:
                                rec_out["dismissed_by"] = bowler_name
                            if kind.lower() == "caught" and caught_names:
                                rec_out["caught_by"] = "|".join(caught_names)
                            # exit at POST state
                            if rec_out["team_score_exit"] is None:
                                rec_out["team_score_exit"] = post_runs
                            if rec_out["team_wkts_exit"] is None:
                                rec_out["team_wkts_exit"] = post_wkts

                    # ---- Bowling (incl. maidens, dot%, boundaries, SR) ----
                    if bowler and bowler.strip().lower() in target_players:
                        bkey = bowler.strip()
                        ensure_player(stats, bkey)
                        rec_bowl = stats[bkey]
                        fielding_team = opponent_team
                        if rec_bowl["team"] is None:
                            rec_bowl["team"] = fielding_team
                            rec_bowl["opponent"] = batting_team
                            rec_bowl["player_of_match"] = 1 if bkey.lower() in potm_list else 0

                        conceded = r_total - byes - legbyes - penalty   # wides/noballs count; byes/leg-byes/penalty don’t
                        if conceded < 0:
                            conceded = 0
                        rec_bowl["runs_conceded"] += conceded

                        rec_bowl["wides_bowled"] += wides
                        rec_bowl["noballs_bowled"] += noballs
                        if r_bat == 4:
                            rec_bowl["fours_conceded"] += 1
                        elif r_bat == 6:
                            rec_bowl["sixes_conceded"] += 1

                        if legal_bowled:
                            rec_bowl["balls_bowled"] += 1
                            over_legal += 1
                            if r_total == 0:
                                rec_bowl["bowler_dot_balls"] += 1

                        over_conceded += conceded

                        for w in d.get("wickets", []) or []:
                            kind_b = (w.get("kind") or "").lower()
                            if kind_b in BOWLER_WICKET_KINDS:
                                rec_bowl["wickets_taken"] += 1
                                rec_bowl["wicket_kinds_bowling"].append(kind_b)

                    # advance inning aggregates
                    inning_runs = post_runs
                    inning_wkts = post_wkts
                    inning_legal_balls = post_balls

                # end of over: maiden if 6 legal + 0 conceded
                if over_bowler and over_bowler.strip().lower() in target_players:
                    bkey = over_bowler.strip()
                    ensure_player(stats, bkey)
                    if stats[bkey]["maidens"] is None:
                        stats[bkey]["maidens"] = 0
                    if over_legal == 6 and over_conceded == 0:
                        stats[bkey]["maidens"] += 1

            # innings end: set exit for not-out batters
            final_runs = inning_runs
            final_wkts = inning_wkts
            for pname, st in list(stats.items()):
                if st["team"] == batting_team:
                    if (st["team_score_entry"] is not None) and (st["team_score_exit"] is None):
                        st["team_score_exit"] = final_runs
                        st["team_wkts_exit"] = final_wkts

        # ---------- outcome / margin ----------
        winner_team = outcome.get("winner")
        by_obj = outcome.get("by", {}) or {}
        by_runs = by_obj.get("runs")
        by_wkts = by_obj.get("wickets")
        result_text = (outcome.get("result") or "").strip().lower()

        if by_runs is not None:
            win_type_code = 1
            win_margin_value = int(by_runs)
        elif by_wkts is not None:
            win_type_code = 2
            win_margin_value = int(by_wkts)
        else:
            win_type_code = 0
            win_margin_value = 0

        # emit rows
        for name, st in stats.items():
            did_bat = (st["team_score_entry"] is not None)  # stricter: appeared as batter at least once
            did_bowl = (st["balls_bowled"] > 0) or (st["wickets_taken"] > 0) or (st["runs_conceded"] > 0) or (st["maidens"] is not None)

            if not (did_bat or did_bowl):
                continue

            toss_win_flag = None
            toss_decision_out = toss_decision if toss_decision in ("bat", "field") else None
            if st["team"] and toss_winner:
                toss_win_flag = 1 if st["team"] == toss_winner else 0

            if st["team"] and winner_team:
                team_won = 1 if st["team"] == winner_team else 0
            elif result_text in ["tie", "no result"]:
                team_won = 0.5
            else:
                team_won = 0

            # batting badges / boundary %
            fifties = hundreds = boundary_pct = None
            if did_bat:
                r = st["runs_scored"]
                fifties  = 1 if (r >= 50 and r < 100) else 0
                hundreds = 1 if (r >= 100) else 0
                if r > 0:
                    boundary_runs = st["fours"] * 4 + st["sixes"] * 6
                    boundary_pct = r2((boundary_runs / r) * 100.0)

            # bowling metrics
            bowler_dot_pct = bowling_sr = economy = maidens_out = None
            if did_bowl:
                maidens_out = st["maidens"] if st["maidens"] is not None else 0
                if st["balls_bowled"] > 0:
                    bowler_dot_pct = r2((st["bowler_dot_balls"] / st["balls_bowled"]) * 100.0)
                    bowling_sr = r2(st["balls_bowled"] / st["wickets_taken"]) if st["wickets_taken"] > 0 else None
                    economy = r2(st["runs_conceded"] / (st["balls_bowled"] / 6.0))

            rows.append({
                "Player": name,
                "Match_ID": match_id,
                "Date": match_date,
                "Opponent": st["opponent"],
                "Event_Name": event_name,
                "Event_Stage": event_stage,
                "Match_Type_Number": match_type_number,
                "Venue": venue_name,
                "Player_of_Match": st["player_of_match"],
                "Toss_Win": toss_win_flag,
                "Toss_Decision": toss_decision_out,
                "Team_Won": team_won,
                "Win_Type": win_type_code,
                "Win_Margin": win_margin_value,

                # batting (blank if didn't bat)
                "Runs_Scored": st["runs_scored"] if did_bat else None,
                "Balls_Faced": st["balls_faced"] if did_bat else None,
                "Dismissal_Kind": st["dismissal_kind"] if did_bat else None,
                "Dismissed_By": st["dismissed_by"] if did_bat else None,
                "Caught_By": st["caught_by"] if did_bat else None,
                "Fours": st["fours"] if did_bat else None,
                "Sixes": st["sixes"] if did_bat else None,
                "Wides_Faced": st["wides_faced"] if did_bat else None,
                "NoBalls_Faced": st["noballs_faced"] if did_bat else None,
                "Batting_Position": st["bat_pos"] if did_bat else None,

                # batting context (blank if didn't bat)
                "Dot_Balls": st["dot_balls"] if did_bat else None,
                "Team_Score_On_Entry": st["team_score_entry"] if did_bat else None,
                "Team_Wickets_On_Entry": st["team_wkts_entry"] if did_bat else None,
                "Team_Score_On_Exit": st["team_score_exit"] if did_bat else None,
                "Team_Wickets_On_Exit": st["team_wkts_exit"] if did_bat else None,
                "Target_Score": st["target_score"] if did_bat else None,
                "OnEntry_Req_RR": st["on_entry_req_rr"] if did_bat else None,
                "OnEntry_Curr_RR": st["on_entry_curr_rr"] if did_bat else None,
                "OnEntry_RR_Diff": st["on_entry_rr_diff"] if did_bat else None,
                "Fifties": fifties,
                "Hundreds": hundreds,
                "Boundary_Pct": boundary_pct,   # % of runs in 4s+6s

                # bowling (blank if didn't bowl)
                "Balls_Bowled": st["balls_bowled"] if did_bowl else None,
                "Runs_Conceded": st["runs_conceded"] if did_bowl else None,
                "Wickets": st["wickets_taken"] if did_bowl else None,
                "Wicket_Kinds": ("|".join(st["wicket_kinds_bowling"]) if st["wicket_kinds_bowling"] else "") if did_bowl else None,
                "Wides_Bowled": st["wides_bowled"] if did_bowl else None,
                "NoBalls_Bowled": st["noballs_bowled"] if did_bowl else None,
                "Fours_Conceded": st["fours_conceded"] if did_bowl else None,
                "Sixes_Conceded": st["sixes_conceded"] if did_bowl else None,
                "Bowler_Dot_Ball_Pct": bowler_dot_pct,
                "Bowling_SR": bowling_sr,
                "Maidens": maidens_out,
                "Economy": economy,
            })

    except Exception as e:
        print(f"[WARN] {fname}: {e}")

# ---------- SAVE ----------
df = pd.DataFrame(rows)

if df.empty:
    print("No rows produced. Double-check roster/JSON inputs.")
else:
    # Position_Mode for rows with Batting_Position
    if "Batting_Position" in df.columns:
        pos_mode = (
            df.dropna(subset=["Batting_Position"])
              .astype({"Batting_Position": "Int64"})
              .groupby("Player")["Batting_Position"]
              .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else None)
              .reset_index()
              .rename(columns={"Batting_Position": "Position_Mode"})
        )
        df = df.merge(pos_mode, on="Player", how="left")
    else:
        df["Position_Mode"] = None

    # ---- Formatting so integers never show ".0", but real decimals keep 2dp ----
    int_cols = [
        "Runs_Scored","Balls_Faced","Fours","Sixes","Wides_Faced","NoBalls_Faced","Batting_Position",
        "Dot_Balls","Team_Score_On_Entry","Team_Wickets_On_Entry","Team_Score_On_Exit","Team_Wickets_On_Exit",
        "Target_Score","Fifties","Hundreds",
        "Balls_Bowled","Runs_Conceded","Wickets","Wides_Bowled","NoBalls_Bowled",
        "Fours_Conceded","Sixes_Conceded","Maidens","Win_Margin","Win_Type","Position_Mode"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].where(df[c].isna(), df[c].astype("Int64"))

    # For decimal metrics: show 2dp if not integer-like, else strip .0
    def strip_trailing_zero(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        try:
            f = float(x)
        except Exception:
            return x
        return int(f) if f.is_integer() else float(Decimal(f).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    float_cols = ["OnEntry_Req_RR","OnEntry_Curr_RR","OnEntry_RR_Diff",
                  "Bowler_Dot_Ball_Pct","Bowling_SR","Economy","Boundary_Pct"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(strip_trailing_zero).astype("object")

    order = [
        "Player","Match_ID","Date","Opponent","Event_Name","Event_Stage","Match_Type_Number","Venue",
        "Player_of_Match","Toss_Win","Toss_Decision","Team_Won","Win_Type","Win_Margin",

        # batting
        "Runs_Scored","Balls_Faced","Dismissal_Kind","Dismissed_By","Caught_By",
        "Fours","Sixes","Wides_Faced","NoBalls_Faced","Batting_Position",

        # batting context
        "Dot_Balls","Team_Score_On_Entry","Team_Wickets_On_Entry",
        "Team_Score_On_Exit","Team_Wickets_On_Exit",
        "Target_Score","OnEntry_Req_RR","OnEntry_Curr_RR","OnEntry_RR_Diff",
        "Fifties","Hundreds","Boundary_Pct",

        # bowling
        "Balls_Bowled","Runs_Conceded","Wickets","Wicket_Kinds","Wides_Bowled","NoBalls_Bowled",
        "Fours_Conceded","Sixes_Conceded","Bowler_Dot_Ball_Pct","Bowling_SR","Maidens","Economy",

        "Position_Mode"
    ]
    for c in order:
        if c not in df.columns:
            df[c] = None

    df = df[order].sort_values(["Player","Date","Match_ID"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df):,} rows → {OUTPUT_CSV}")
