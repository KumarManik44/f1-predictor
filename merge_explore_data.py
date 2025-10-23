import pandas as pd
import os

print("Loading all collected data...\n")

# Load race results
race_2020_2024 = pd.read_csv('race_results_2020_2024.csv')
race_2025 = pd.read_csv('race_results_2025.csv')
all_races = pd.concat([race_2020_2024, race_2025], ignore_index=True)

print(f"✅ Total race records: {len(all_races)}")
print(f"   - 2020-2024: {len(race_2020_2024)}")
print(f"   - 2025: {len(race_2025)}")

# Load qualifying results
quali_2020_2024 = pd.read_csv('qualifying_results_2020_2024.csv')
quali_2025 = pd.read_csv('qualifying_results_2025.csv')
all_quali = pd.concat([quali_2020_2024, quali_2025], ignore_index=True)

print(f"\n✅ Total qualifying records: {len(all_quali)}")
print(f"   - 2020-2024: {len(quali_2020_2024)}")
print(f"   - 2025: {len(quali_2025)}")

# Explore race data structure
print(f"\n📊 Race Data Columns:")
print(all_races.columns.tolist())

print(f"\n📊 Qualifying Data Columns:")
print(all_quali.columns.tolist())

# Check unique values
print(f"\n📈 Data Summary:")
print(f"   - Total seasons: {all_races['season'].nunique()}")
print(f"   - Total races: {len(all_races.groupby(['season', 'round']))}")
print(f"   - Unique drivers: {all_races['driverCode'].nunique()}")
print(f"   - Unique constructors: {all_races['constructorName'].nunique()}")

# Show 2025 driver standings (current season)
print(f"\n🏆 2025 Driver Standings (so far):")
driver_standings_2025 = race_2025.groupby('driverCode')['points'].sum().sort_values(ascending=False)
print(driver_standings_2025.head(10))

# Show sample merged view
print(f"\n📋 Sample Race Data:")
print(all_races[['season', 'round', 'driverCode', 'constructorName', 'position', 'points']].head(10))

# Save combined datasets
all_races.to_csv('all_race_results_2020_2025.csv', index=False)
all_quali.to_csv('all_qualifying_results_2020_2025.csv', index=False)

print(f"\n💾 Combined datasets saved:")
print(f"   - all_race_results_2020_2025.csv")
print(f"   - all_qualifying_results_2020_2025.csv")

print(f"\n✅ Data exploration complete! Ready for feature engineering.")
