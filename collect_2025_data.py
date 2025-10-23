import fastf1
import pandas as pd
import os
import time
from fastf1.ergast import Ergast

# Configure API and cache
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Initialize Ergast API
ergast = Ergast()

print("Collecting 2025 season data...\n")

# Collect race results
all_race_results = []
all_qualifying_results = []

year = 2025
print(f"Fetching {year} season data...")

# Get all races scheduled for 2025
races = ergast.get_race_schedule(year)
print(f"Total races scheduled in 2025: {len(races)}\n")

for race_round in range(1, len(races) + 1):
    try:
        # Get race results
        race_results = ergast.get_race_results(season=year, round=race_round)

        if race_results.content and len(race_results.content) > 0:
            race_df = race_results.content[0]
            race_df['season'] = year
            race_df['round'] = race_round
            all_race_results.append(race_df)
            print(f"  ✅ Race Round {race_round}: {len(race_df)} drivers")
        else:
            print(f"  ⚠️  Race Round {race_round}: Not completed yet")

        time.sleep(1)

        # Get qualifying results
        quali_results = ergast.get_qualifying_results(season=year, round=race_round)

        if quali_results.content and len(quali_results.content) > 0:
            quali_df = quali_results.content[0]
            quali_df['season'] = year
            quali_df['round'] = race_round
            all_qualifying_results.append(quali_df)
            print(f"  ✅ Quali Round {race_round}: {len(quali_df)} drivers")
        else:
            print(f"  ⚠️  Quali Round {race_round}: Not available yet")

        time.sleep(1)

    except Exception as e:
        print(f"  ❌ Round {race_round}: Error - {e}")
        if "Too Many Requests" in str(e):
            print("     ⏳ Rate limited. Waiting 5 seconds...")
            time.sleep(5)
        continue

# Save race results
if len(all_race_results) > 0:
    race_dataset_2025 = pd.concat(all_race_results, ignore_index=True)
    race_output = 'race_results_2025.csv'
    race_dataset_2025.to_csv(race_output, index=False)
    print(f"\n✅ 2025 Race data saved!")
    print(f"📊 Completed races: {len(all_race_results)}")
    print(f"📊 Total records: {len(race_dataset_2025)}")
    print(f"💾 Saved to: {race_output}")
else:
    print(f"\n⚠️  No completed 2025 races found yet")

# Save qualifying results
if len(all_qualifying_results) > 0:
    quali_dataset_2025 = pd.concat(all_qualifying_results, ignore_index=True)
    quali_output = 'qualifying_results_2025.csv'
    quali_dataset_2025.to_csv(quali_output, index=False)
    print(f"\n✅ 2025 Qualifying data saved!")
    print(f"📊 Completed qualifyings: {len(all_qualifying_results)}")
    print(f"💾 Saved to: {quali_output}")
else:
    print(f"\n⚠️  No completed 2025 qualifying sessions found yet")

# Show sample of recent races
if len(all_race_results) > 0:
    print(f"\n🏁 Most Recent 2025 Races:")
    recent_races = race_dataset_2025.groupby('round').first().tail(3)[['raceName']]
    print(recent_races)
