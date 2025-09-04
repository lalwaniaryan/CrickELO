# app.py
# Wires added:
# - Pass INPUT_FILE_OVERRIDE to subprocess so recompute uses the uploaded/selected CSV
# - After recompute, clear cache & rerun so tables refresh immediately

import os, subprocess
import pandas as pd
import streamlit as st
import plotly.express as px

RAW_FILE = "player_match_by_match_stats.csv"
MATCH_CSV = "match_scores.csv"
PLAYER_CSV = "player_elo_ratings.csv"

st.set_page_config(page_title="ODI Elo Rankings", layout="wide")

# -------- Sidebar: upload + knobs --------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload player_match_by_match_stats.csv", type=["csv"])
if uploaded:
    # Save under a deterministic name (so you can share a link & reproduce)
    RAW_FILE = "player_match_by_match_stats_uploaded.csv"
    with open(RAW_FILE, "wb") as f:
        f.write(uploaded.getvalue())
    st.sidebar.success(f"Uploaded file saved as {RAW_FILE}")

st.sidebar.markdown("### Hyperparameters")
perf_scale = st.sidebar.slider("PERF_SCALE (logistic width)", 0.25, 2.0, 0.75, 0.05)
k_base     = st.sidebar.slider("K_BASE", 5.0, 60.0, 20.0, 1.0)
z_clip     = st.sidebar.slider("Z_CLIP (clip z-scores)", 2.0, 6.0, 4.0, 0.5)
recompute  = st.sidebar.button("Recompute Elo")

# -------- Recompute pipeline --------
def run_pipeline(active_file: str):
    env = os.environ.copy()
    # Wire: tell the Elo script which CSV & tuning to use
    env["INPUT_FILE_OVERRIDE"] = active_file
    env["PERF_SCALE_OVERRIDE"] = str(perf_scale)
    env["K_BASE_OVERRIDE"]     = str(k_base)
    env["Z_CLIP_OVERRIDE"]     = str(z_clip)

    result = subprocess.run(["python", "elo_weighted_rank.py"],
                            env=env, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Pipeline failed:")
        st.code(result.stderr)
        return False
    st.success("Recomputed Elo successfully.")
    if result.stdout.strip():
        st.code(result.stdout)
    return True

if recompute:
    ok = run_pipeline(RAW_FILE)
    if ok:
        # Clear cached CSVs so fresh data loads
        load_csv_safe.clear()
        st.experimental_rerun()

# -------- Load outputs --------
@st.cache_data
def load_csv_safe(path):
    if not os.path.exists(path): return None
    return pd.read_csv(path)

df_players = load_csv_safe(PLAYER_CSV)
df_matches = load_csv_safe(MATCH_CSV)

st.title("🏏 ODI Elo Rankings — Weighted by Feature Importance")

if df_players is None or df_matches is None:
    st.warning("Outputs not found. Click **Recompute Elo** in the sidebar (ensure raw CSV is present).")
    st.stop()

# -------- Leaderboard --------
st.subheader("Leaderboard")
with st.expander("About this table", expanded=False):
    st.markdown("""
- **Elo**: Solo Elo vs. a 1500 baseline, updated each match using your weighted performance.
- **Matches_Played**: Appearances in the dataset.
- Use the filter below to reduce small-sample noise.
""")

min_games = st.slider("Minimum matches to display", 1, int(df_players["Matches_Played"].max()), 5, 1)
lb = (df_players[df_players["Matches_Played"] >= min_games]
      .sort_values(["Elo","Matches_Played"], ascending=[False, False])
      .reset_index(drop=True))
st.dataframe(lb, use_container_width=True)

# -------- Player Explorer --------
st.subheader("Player explorer")
players = sorted(df_players["Player"].unique().tolist())
sel_player = st.selectbox("Select a player", options=players, index=0 if players else None)

if sel_player:
    pm = df_matches[df_matches["Player"] == sel_player].copy()
    pm["DateParsed"] = pd.to_datetime(pm["Date"], errors="coerce")

    col1, col2 = st.columns([2,1], gap="large")

    with col1:
        st.markdown("**Elo over time**")
        fig = px.line(pm.sort_values("DateParsed"), x="DateParsed", y="Elo_After",
                      markers=True, hover_data=["Match_ID","Opponent","Venue","Runs_Scored","Balls_Faced"])
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Elo")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Recent matches**")
        cols = [c for c in ["Date","Match_ID","Opponent","Venue","Runs_Scored","Balls_Faced","Fours","Sixes","ObservedScore","K_used","Elo_After"] if c in pm.columns]
        st.dataframe(pm.sort_values("DateParsed", ascending=False)[cols].head(12), use_container_width=True)

    st.markdown("**Performance distribution** (WeightedPerf)")
    if "WeightedPerf" in pm.columns:
        hist = px.histogram(pm, x="WeightedPerf", nbins=25)
        hist.update_layout(height=300, xaxis_title="WeightedPerf", yaxis_title="Count")
        st.plotly_chart(hist, use_container_width=True)

# -------- Diagnostics --------
st.subheader("Diagnostics")
present = [c for c in ["Runs_Scored","Balls_Faced","Fours","Sixes","Team_Won","Win_Margin","ObservedScore","K_used","WeightedPerf"] if c in df_matches.columns]
if present:
    corr = df_matches[present].corr(numeric_only=True)
    st.markdown("**Correlations (numeric)**")
    st.dataframe(corr.get("WeightedPerf", pd.Series(dtype=float)).sort_values(ascending=False).to_frame(), use_container_width=True)

st.caption("Tip: validate hypotheses with AUC/LogLoss offline, then port to Next.js + FastAPI for production.")
