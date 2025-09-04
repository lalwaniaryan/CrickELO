WORKOUT_SYSTEM = """You are SmartCoach AI, a realistic fitness and nutrition planner. 
Keep workouts safe, practical, and easy to follow for beginners.
Always explain progression and keep the plan aligned with healthy, sustainable weight loss."""

WORKOUT_USER = """TASK: Create a combined workout + meal plan.

User Info:
- Age: 22
- Status: Just started working (beginner fitness level)
- Goal: Lose 3kg in 3 months
- Workout time available: 2 hours per day
- Equipment: Dumbbells + Bodyweight
- Duration: 12 weeks (3 months)

Requirements:
1. Workout Plan:
   - Structure as a weekly schedule.
   - 4–5 sessions per week, ≤120 minutes/session.
   - Include warm-up, main exercises, sets, reps, rest.
   - Explain weekly progression and add recovery days.

2. Meal Plan:
   - Daily calorie target appropriate for sustainable fat loss (~0.5–1 kg/week).
   - Balance protein, carbs, fats (protein ~1.6–2 g/kg body weight).
   - Give a 1-day example meal plan and grocery list.
   - Keep meals simple, affordable, and easy to prep.

OUTPUT FORMAT:
- "Workout Plan": structured weekly program with progression notes.
- "Meal Plan": daily meals, macros, grocery list.
"""
