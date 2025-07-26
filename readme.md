# Cricket Elo-Based Player Ranking System

## Project Goal
Develop an Elo-based player ranking system for cricket, focusing on both batting and bowling performance.

## Understanding the Elo Rating System
- The Elo system ranks players based on match outcomes.
- **Initial Rating**: All players begin with a base score (e.g., 1500).
- **Rating Updates**: After each match:
  - Performers gain or lose points based on their impact and the result.
  - Contextual factors (e.g., strength of opposition, match pressure) will adjust the weight of performance.

## Core Performance Variables

### Batting Metrics
- Runs scored
- Strike rate (SR)
- Dismissal status (out/not out)
- 50s, 100s
- Boundary count (4s, 6s)
- Phase-specific performance (Powerplay, Middle, Death overs)
- Leverage-adjusted contribution (e.g., performance under pressure)

### Bowling Metrics
- Wickets taken
- Runs conceded
- Dot ball %, economy rate
- Phase effectiveness
- Type and difficulty of dismissals
- Pressure metrics (e.g., bowling during tight situations)

## Match Context Variables
- Match format (Test, ODI, T20I)
- Match stage (league, knockout, final)
- Opposition strength (via team/player Elo)
- Venue conditions (pitch rating, weather, home/away/neutral)
- Toss outcome and decision
- Target score (for chasing scenarios)
- Entry point conditions (score, overs remaining, required run rate)
- Win probability swing during contribution
- Batting/bowling partner performance

## Brief Intro to Cricket (for Context)
Cricket is a team sport played between two teams of 11 players each. The game revolves around scoring runs by:
- Running between wickets after hitting the ball.
- 4 runs for hitting the ball past the boundary after bouncing.
- 6 runs for clearing the boundary on the full.

### Key Roles
- **Bowler**: Delivers the ball aiming to get the batter out or restrict runs.
- **Batsman**: Tries to score runs by hitting the ball effectively while avoiding dismissal.

## Detailed Metrics (Batting and Bowling)

[Refer to full description for Batsman-centric, Bowler-centric, Match Context, Situational Pressure, Delivery Attributes, and Fatigue Factors.]

## ELO LOGIC
- **Initialise Ratings**: Starting rating of 1500 per format.
- **Differentiate for Bowlers and Batsman**.
- **Focus on Core Variables First**.
- **Calculate the weighted score based on performance across variables.**