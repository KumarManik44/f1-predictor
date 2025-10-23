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

# Initialize Ergast API for historical data
ergast = Ergast()

print("Collecting historical race data (2020-2024)...\n")

# Collect race results for each season
all_race_results = []

for year in range(2020, 2025):  # 2020 to 2024
    print(f"Fetching {year} season data...")

    # Get all races for this year
    races = ergast.get_race_schedule(year)

    for race_round in range(1, len(races) + 1):
        try:
            # Get race results
            race_results = ergast.get_race_results(season=year, round=race_round)

            # Check if we got data (it's a list, not a DataFrame)
            if race_results.content and len(race_results.content) > 0:
                race_df = race_results.content[0]  # Get the first (and only) race
                race_df['season'] = year
                race_df['round'] = race_round
                all_race_results.append(race_df)
                print(f"  ✅ Round {race_round}: {len(race_df)} drivers")
            else:
                print(f"  ⚠️  Round {race_round}: No data available")

            # Add delay to avoid rate limiting (1 second between requests)
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Round {race_round}: Error - {e}")
            # If rate limited, wait longer
            if "Too Many Requests" in str(e):
                print("     ⏳ Rate limited. Waiting 5 seconds...")
                time.sleep(5)
            continue

# Check if we collected any data
if len(all_race_results) == 0:
    print("\n❌ No data collected. Please try again later or check API status.")
else:
    # Combine all race results
    full_dataset = pd.concat(all_race_results, ignore_index=True)

    # Save to CSV
    output_file = 'race_results_2020_2024.csv'
    full_dataset.to_csv(output_file, index=False)

    print(f"\n✅ Data collection complete!")
    print(f"📊 Total races collected: {len(all_race_results)}")
    print(f"📊 Total records: {len(full_dataset)}")
    print(f"💾 Saved to: {output_file}")

    # Display sample
    print(f"\nSample data (first 10 rows):")
    print(full_dataset[['season', 'round', 'driverCode', 'constructorName', 'position', 'points']].head(10))
