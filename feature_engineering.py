import pandas as pd
import numpy as np

print("Loading combined datasets...\n")

# Load data
races = pd.read_csv('all_race_results_2020_2025.csv')
quali = pd.read_csv('all_qualifying_results_2020_2025.csv')

print(f"✅ Loaded {len(races)} race records")
print(f"✅ Loaded {len(quali)} qualifying records\n")

# Sort by season and round
races = races.sort_values(['season', 'round', 'position']).reset_index(drop=True)
quali = quali.sort_values(['season', 'round', 'position']).reset_index(drop=True)

# Create unique race identifier
races['race_id'] = races['season'].astype(str) + '_' + races['round'].astype(str)
quali['race_id'] = quali['season'].astype(str) + '_' + quali['round'].astype(str)

print("Creating features...\n")

# ===== FEATURE 1: Grid Position (from qualifying) =====
print("1️⃣ Creating grid position feature...")
quali_grid = quali[['race_id', 'driverCode', 'position']].copy()
quali_grid.rename(columns={'position': 'grid_position'}, inplace=True)
races = races.merge(quali_grid, on=['race_id', 'driverCode'], how='left')

# ===== FEATURE 2: Driver Form - Last 5 Races Points =====
print("2️⃣ Creating driver form (last 5 races avg points)...")
races['driver_last5_avg_points'] = 0.0

for idx, row in races.iterrows():
    # Get driver's last 5 races before this one
    prev_races = races[
        (races['driverCode'] == row['driverCode']) &
        (races['season'] <= row['season']) &
        ((races['season'] < row['season']) | (races['round'] < row['round']))
        ].tail(5)

    if len(prev_races) > 0:
        races.at[idx, 'driver_last5_avg_points'] = prev_races['points'].mean()

# ===== FEATURE 3: Driver Last 5 Races Avg Position =====
print("3️⃣ Creating driver avg finish position (last 5 races)...")
races['driver_last5_avg_position'] = 0.0

for idx, row in races.iterrows():
    prev_races = races[
        (races['driverCode'] == row['driverCode']) &
        (races['season'] <= row['season']) &
        ((races['season'] < row['season']) | (races['round'] < row['round']))
        ].tail(5)

    if len(prev_races) > 0:
        races.at[idx, 'driver_last5_avg_position'] = prev_races['position'].mean()

# ===== FEATURE 4: Constructor Season Points =====
print("4️⃣ Creating constructor season points...")
races['constructor_season_points'] = 0.0

for idx, row in races.iterrows():
    # Get constructor's points in current season before this race
    season_races = races[
        (races['constructorName'] == row['constructorName']) &
        (races['season'] == row['season']) &
        (races['round'] < row['round'])
        ]

    if len(season_races) > 0:
        races.at[idx, 'constructor_season_points'] = season_races['points'].sum()

# ===== FEATURE 5: Driver Season Points =====
print("5️⃣ Creating driver season points...")
races['driver_season_points'] = 0.0

for idx, row in races.iterrows():
    # Get driver's points in current season before this race
    season_races = races[
        (races['driverCode'] == row['driverCode']) &
        (races['season'] == row['season']) &
        (races['round'] < row['round'])
        ]

    if len(season_races) > 0:
        races.at[idx, 'driver_season_points'] = season_races['points'].sum()

# ===== FEATURE 6: Podium Finish (Target Variable) =====
print("6️⃣ Creating target variable (podium finish)...")
races['podium_finish'] = (races['position'] <= 3).astype(int)

# ===== FEATURE 7: Race Winner (Alternative Target) =====
races['race_winner'] = (races['position'] == 1).astype(int)

print("\n✅ Feature engineering complete!\n")

# Show feature summary
feature_cols = ['grid_position', 'driver_last5_avg_points', 'driver_last5_avg_position',
                'constructor_season_points', 'driver_season_points', 'podium_finish', 'race_winner']

print("📊 Feature Summary:")
print(races[feature_cols].describe())

# Save dataset with features
output_file = 'f1_dataset_with_features.csv'
races.to_csv(output_file, index=False)
print(f"\n💾 Dataset with features saved to: {output_file}")

# Show sample
print(f"\n📋 Sample data with features:")
display_cols = ['season', 'round', 'driverCode', 'constructorName', 'grid_position',
                'driver_last5_avg_points', 'driver_season_points', 'position', 'podium_finish']
print(races[display_cols].head(20))
