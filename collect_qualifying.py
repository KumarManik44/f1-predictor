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

print("Collecting qualifying data (2020-2024)...\n")

# Collect qualifying results for each season
all_qualifying_results = []

for year in range(2020, 2025):  # 2020 to 2024
    print(f"Fetching {year} qualifying data...")

    # Get all races for this year
    races = ergast.get_race_schedule(year)

    for race_round in range(1, len(races) + 1):
        try:
            # Get qualifying results
            quali_results = ergast.get_qualifying_results(season=year, round=race_round)

            if quali_results.content and len(quali_results.content) > 0:
                quali_df = quali_results.content[0]
                quali_df['season'] = year
                quali_df['round'] = race_round
                all_qualifying_results.append(quali_df)
                print(f"  ✅ Round {race_round}: {len(quali_df)} drivers")
            else:
                print(f"  ⚠️  Round {race_round}: No qualifying data")

            # Rate limiting delay
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Round {race_round}: Error - {e}")
            if "Too Many Requests" in str(e):
                print("     ⏳ Rate limited. Waiting 5 seconds...")
                time.sleep(5)
            continue

# Check if we collected any data
if len(all_qualifying_results) == 0:
    print("\n❌ No qualifying data collected.")
else:
    # Combine all qualifying results
    full_quali_dataset = pd.concat(all_qualifying_results, ignore_index=True)

    # Save to CSV
    output_file = 'qualifying_results_2020_2024.csv'
    full_quali_dataset.to_csv(output_file, index=False)

    print(f"\n✅ Qualifying data collection complete!")
    print(f"📊 Total races collected: {len(all_qualifying_results)}")
    print(f"📊 Total records: {len(full_quali_dataset)}")
    print(f"💾 Saved to: {output_file}")

    # Display sample
    print(f"\nSample qualifying data (first 10 rows):")
    print(
        full_quali_dataset[['season', 'round', 'driverCode', 'constructorName', 'position', 'Q1', 'Q2', 'Q3']].head(10))
