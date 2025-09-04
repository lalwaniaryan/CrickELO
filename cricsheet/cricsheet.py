import os
import json
import pandas as pd
from tqdm import tqdm

# --- 1. Configuration - SET YOUR FILE PATHS HERE ---
PLAYER_LIST_PATH = 'ODI weighted.csv'
PLAYER_COLUMN_NAME = 'Player'
DATA_FOLDER = 'odis_json'
OUTPUT_FILENAME = 'player_match_by_match_stats_sorted.csv'

# --- 2. Load Target Players ---
try:
    if PLAYER_LIST_PATH.endswith('.csv'):
        players_df = pd.read_csv(PLAYER_LIST_PATH)
    else:
        players_df = pd.read_excel(PLAYER_LIST_PATH)
    target_players = set(players_df[PLAYER_COLUMN_NAME].str.lower().str.strip())
    print(f"Successfully loaded {len(target_players)} players from '{PLAYER_LIST_PATH}'.")
except Exception as e:
    print(f"Error loading player list: {e}")
    exit()

# --- 3. Process Match Files ---
all_match_records = []

if not os.path.isdir(DATA_FOLDER):
    print(f"ERROR: Data folder '{DATA_FOLDER}' not found.")
    exit()

json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
print(f"\nProcessing {len(json_files)} match files...")

for filename in tqdm(json_files, desc="Processing Matches"):
    filepath = os.path.join(DATA_FOLDER, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            match_data = json.load(f)

        # --- Extract Match-Level Information ---
        info = match_data.get('info', {})
        if info.get('match_type') != 'ODI':
            continue

        match_date = info.get('dates', [None])[0]
        teams = info.get('teams', [])
        venue = info.get('venue', 'N/A')
        event_info = info.get('event', {})
        event_name = event_info.get('name', 'Bilateral Series')
        match_number = event_info.get('match_number')
        stage = event_info.get('stage', 'N/A')
        player_of_match_list = [name.lower().strip() for name in info.get('player_of_match', [])]
        
        match_player_stats = {}

        for inning in match_data.get('innings', []):
            team_batting = inning.get('team')
            opponent = teams[1] if teams[0] == team_batting else teams[0]
            
            for over in inning.get('overs', []):
                for delivery in over.get('deliveries', []):
                    batter = delivery.get('batter', '').lower().strip()
                    bowler = delivery.get('bowler', '').lower().strip()
                    
                    for Player in [batter, bowler]:
                        if Player in target_players and Player not in match_player_stats:
                            match_player_stats[Player] = {
                                'runs_scored': 0, 'balls_faced': 0, 'fours': 0, 'sixes': 0,
                                'balls_bowled': 0, 'runs_conceded': 0, 'wickets_taken': 0,
                                'is_dismissed': 0,
                                'dismissal_kind': 'Not Out'  # <-- NEW: Default dismissal status
                            }

                    # Batting stats
                    if batter in target_players:
                        runs = delivery.get('runs', {})
                        batter_runs = runs.get('batter', 0)
                        match_player_stats[batter]['runs_scored'] += batter_runs
                        match_player_stats[batter]['balls_faced'] += 1
                        if batter_runs == 4: match_player_stats[batter]['fours'] += 1
                        if batter_runs == 6: match_player_stats[batter]['sixes'] += 1

                    # Bowling stats
                    if bowler in target_players:
                        match_player_stats[bowler]['balls_bowled'] += 1
                        match_player_stats[bowler]['runs_conceded'] += delivery.get('runs', {}).get('total', 0)
                        if 'wicket' in delivery and delivery['wicket'].get('kind') not in ['run out', 'retired hurt', 'obstructing the field']:
                            match_player_stats[bowler]['wickets_taken'] += 1

                    # Dismissal info
                    if 'wicket' in delivery:
                        wicket_info = delivery['wicket']
                        player_out = wicket_info.get('player_out', '').lower().strip()
                        if player_out in target_players:
                             match_player_stats[player_out]['is_dismissed'] = 1
                             # <-- NEW: Capture how the player got out
                             match_player_stats[player_out]['dismissal_kind'] = wicket_info.get('kind', 'N/A')

        # --- Create a Record for Each Player in the Match ---
        for Player, stats in match_player_stats.items():
            record = {
                'Player': Player.title(),
                'Match_ID': filename.replace('.json', ''),
                'Date': match_date,
                'Opponent': opponent,
                'Event_Name': event_name,
                'Event_Stage': stage,
                'Player_of_Match': 1 if Player in player_of_match_list else 0,
                'Runs_Scored': stats['runs_scored'],
                'Balls_Faced': stats['balls_faced'],
                'Fours': stats['fours'],
                'Sixes': stats['sixes'],
                'Is_Dismissed': stats['is_dismissed'],
                'Dismissal_Kind': stats['dismissal_kind'], # <-- NEW: Add dismissal kind to record
                'Balls_Bowled': stats['balls_bowled'],
                'Runs_Conceded': stats['runs_conceded'],
                'Wickets_Taken': stats['wickets_taken']
            }
            all_match_records.append(record)

    except Exception as e:
        print(f"\nAn error occurred with file {filename}: {e}. Skipping.")

# --- 4. Create, Sort, and Save the Final DataFrame ---
print("\nCreating DataFrame, sorting, and saving to CSV...")
df = pd.DataFrame(all_match_records)

# <-- NEW: Sort the data first by player, then by date
df.sort_values(by=['Player', 'Date'], inplace=True)

# Reorder columns for better readability
df = df[[
    'Player', 'Match_ID', 'Date', 'Opponent', 'Event_Name', 'Event_Stage', 'Player_of_Match',
    'Runs_Scored', 'Balls_Faced', 'Dismissal_Kind', 'Fours', 'Sixes', 'Is_Dismissed',
    'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken'
]]

df.to_csv(OUTPUT_FILENAME, index=False)

print(f"✅ Success! Your sorted, detailed table has been saved to '{OUTPUT_FILENAME}'.")