# app.py
# Wires:
# - Pass INPUT_FILE_OVERRIDE to Elo script
# - Read sidebar hyperparams via env
# - Clear cache after recompute, then rerun

import os, subprocess
import pandas as pd
import streamlit as st
import plotly.express as px

RAW_FILE   = "player_match_by_match_stats_sorted.csv"
MATCH_CSV  = "match_scores.csv"
PLAYER_CSV = "player_elo_ratings.csv"

st.set_page_config(page_title="ODI Elo Rankings", layout="wide")

# ---------------------------
# Cached CSV loader (define BEFORE we call .clear())
# ---------------------------
@st.cache_data
def load_csv_safe(path: str):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

# ---------------------------
# Sidebar: upload + knobs
# ---------------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload player_match_by_match_stats.csv",
    type=["csv"]
)
if uploaded:
    RAW_FILE = "player_match_by_match_stats_uploaded.csv"
    with open(RAW_FILE, "wb") as f:
        f.write(uploaded.getvalue())
    st.sidebar.success(f"Uploaded file saved as {RAW_FILE}")

st.sidebar.markdown("### Hyperparameters")

# Existing knobs
perf_scale = st.sidebar.slider("PERF_SCALE (logistic width)", 0.10, 2.0, 0.75, 0.05)
k_base     = st.sidebar.slider("K_BASE", 5.0, 60.0, 20.0, 1.0)
z_clip     = st.sidebar.slider("Z_CLIP (clip z-scores)", 2.0, 8.0, 4.0, 0.5)

# NEW knobs to unlock rating range
elo_scale    = st.sidebar.slider("ELO_SCALE (rating spread)", 300.0, 1600.0, 900.0, 50.0)
perf_stretch = st.sidebar.slider("PERF_STRETCH (separation)", 1.0, 10.0, 3.0, 0.25)

st.sidebar.caption("ELO_START is fixed at 1500. Increase ELO_SCALE / PERF_STRETCH to allow higher peaks.")

# ---------------------------
# Recompute pipeline (single, final)
# ---------------------------
def run_pipeline(active_file: str) -> bool:
    with st.spinner("Computing Elo…"):
        env = os.environ.copy()
        env["INPUT_FILE_OVERRIDE"]        = active_file
        env["PERF_SCALE_OVERRIDE"]        = str(perf_scale)
        env["K_BASE_OVERRIDE"]            = str(k_base)
        env["Z_CLIP_OVERRIDE"]            = str(z_clip)
        env["ELO_SCALE_OVERRIDE"]         = str(elo_scale)
        env["PERF_STRETCH_OVERRIDE"]      = str(perf_stretch)

        result = subprocess.run(
            ["python", "elo_weighted_rank.py"],
            env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("Pipeline failed")
            if result.stderr.strip():
                st.code(result.stderr)
            if result.stdout.strip():
                st.code(result.stdout)
            return False

        st.success("Recomputed Elo successfully")
        if result.stdout.strip():
            st.code(result.stdout)
        return True

if st.sidebar.button("Recompute Elo"):
    ok = run_pipeline(RAW_FILE)
    if ok:
        load_csv_safe.clear()

        # Streamlit version compatibility: new versions use st.rerun()
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# ---------------------------
# Load outputs
# ---------------------------
df_players = load_csv_safe(PLAYER_CSV)
df_matches = load_csv_safe(MATCH_CSV)

st.title("🏏 ODI Elo Rankings — Weighted by Feature Importance")

if df_players is None or df_matches is None:
    st.warning("Outputs not found. Click **Recompute Elo** in the sidebar (ensure raw CSV is present).")
    st.stop()

# ---------------------------
# Leaderboard
# ---------------------------
st.subheader("Leaderboard")
with st.expander("About this table", expanded=False):
    st.markdown("""
- **Elo**: Player rating starting at 1500, updated each match using weighted performance.
- **Matches_Played**: Appearances in the dataset.
- Tip: increase **ELO_SCALE** and/or **PERF_STRETCH** if ratings cluster too tightly (e.g., stuck ~1600).
""")

max_played = int(df_players["Matches_Played"].max()) if len(df_players) else 1
min_games = st.slider("Minimum matches to display", 1, max(1, max_played), min(5, max_played), 1)

lb = (df_players[df_players["Matches_Played"] >= min_games]
      .sort_values(["Elo","Matches_Played"], ascending=[False, False])
      .reset_index(drop=True))
st.dataframe(lb, use_container_width=True)

# ---------------------------
# Player Explorer
# ---------------------------
st.subheader("Player explorer")
players = sorted(df_players["Player"].unique().tolist())
sel_player = st.selectbox("Select a player", options=players, index=0 if players else None)

if sel_player:
    pm = df_matches[df_matches["Player"] == sel_player].copy()
    pm["DateParsed"] = pd.to_datetime(pm["Date"], errors="coerce")

    col1, col2 = st.columns([2,1], gap="large")

    with col1:
        st.markdown("**Elo over time**")
        fig = px.line(
            pm.sort_values("DateParsed"),
            x="DateParsed", y="Elo_After",
            markers=True,
            hover_data=[c for c in ["Match_ID","Opponent","Venue","Runs_Scored","Balls_Faced","ObservedScore","K_used"] if c in pm.columns]
        )
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Elo")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Recent matches**")
        cols = [c for c in [
            "Date","Match_ID","Opponent","Venue","Runs_Scored","Balls_Faced",
            "Fours","Sixes","WeightedPerf","ObservedScore","Elo_Before","Elo_Expected","K_used","Elo_After"
        ] if c in pm.columns]
        st.dataframe(pm.sort_values("DateParsed", ascending=False)[cols].head(12), use_container_width=True)

    st.markdown("**Performance distribution** (WeightedPerf)")
    if "WeightedPerf" in pm.columns:
        hist = px.histogram(pm, x="WeightedPerf", nbins=25)
        hist.update_layout(height=300, xaxis_title="WeightedPerf", yaxis_title="Count")
        st.plotly_chart(hist, use_container_width=True)

# ---------------------------
# Diagnostics
# ---------------------------
st.subheader("Diagnostics")
present = [c for c in [
    "Runs_Scored","Balls_Faced","Fours","Sixes","Team_Won","Win_Margin",
    "ObservedScore","K_used","WeightedPerf","Elo_After"
] if c in df_matches.columns]
if present:
    corr = df_matches[present].corr(numeric_only=True)
    target_col = "Elo_After" if "Elo_After" in corr.columns else present[0]
    st.markdown("**Correlations (numeric)**")
    st.dataframe(
        corr.get(target_col, pd.Series(dtype=float)).sort_values(ascending=False).to_frame(),
        use_container_width=True
    )

st.caption("Tip: If ratings still cluster, inspect WeightedPerf distribution; separation is driven by PERF_STRETCH and PERF_SCALE.")
