import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score

print("Loading trained model and data...\n")

# Load the feature dataset
df = pd.read_csv('f1_dataset_with_features.csv')

# Feature columns (same as training)
feature_columns = [
    'grid_position',
    'driver_last5_avg_points',
    'driver_last5_avg_position',
    'constructor_season_points',
    'driver_season_points'
]

# Prepare training data (2020-2024)
train_data = df[df['season'] < 2025].dropna(subset=feature_columns + ['podium_finish'])
X_train = train_data[feature_columns]
y_train = train_data['podium_finish']

# Train model
print("🤖 Training model on 2020-2024 data...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
print("✅ Model ready!\n")

# Get Round 20 data (next race)
next_race = df[(df['season'] == 2025) & (df['round'] == 20)].copy()

if len(next_race) == 0:
    print("⚠️  Round 20 data not available yet. Qualifying hasn't happened.")
    print("\nTo predict Round 20, we need:")
    print("  1. Qualifying results (grid positions)")
    print("  2. Driver stats up to Round 19")
    print("\nLet me show you how to predict using current driver stats...")

    # Get latest driver stats from Round 19
    print("\n📊 Current 2025 Driver Stats (after Round 19):")

    latest_stats = df[df['season'] == 2025].groupby('driverCode').agg({
        'driver_last5_avg_points': 'last',
        'driver_last5_avg_position': 'last',
        'driver_season_points': 'last',
        'constructorName': 'last'
    }).reset_index()

    latest_stats = latest_stats.sort_values('driver_season_points', ascending=False)
    print(latest_stats.head(15))

    print("\n💡 Once Round 20 qualifying happens, update the grid positions and run prediction!")

else:
    print(f"🏁 Predicting Round 20 Winner!\n")

    # Check if we have grid positions
    if next_race['grid_position'].isna().all():
        print("⚠️  Grid positions not available yet (qualifying not completed)")
    else:
        # Prepare features for prediction
        X_next = next_race[feature_columns].dropna()

        if len(X_next) > 0:
            # Make predictions
            podium_proba = model.predict_proba(X_next)[:, 1]

            # Create results DataFrame
            results = next_race.loc[X_next.index, ['driverCode', 'constructorName', 'grid_position']].copy()
            results['podium_probability'] = podium_proba
            results['win_probability'] = podium_proba  # Approximate (top of podium probabilities)

            # Sort by probability
            results = results.sort_values('podium_probability', ascending=False)

            print("📊 Round 20 Predictions:\n")
            print(results.to_string(index=False))

            print(f"\n🏆 PREDICTED WINNER: {results.iloc[0]['driverCode']}")
            print(f"   Constructor: {results.iloc[0]['constructorName']}")
            print(f"   Win Probability: {results.iloc[0]['win_probability']:.1%}")

            print(f"\n🥈 PREDICTED PODIUM:")
            for i in range(min(3, len(results))):
                driver = results.iloc[i]
                print(f"   P{i + 1}: {driver['driverCode']} ({driver['podium_probability']:.1%})")
        else:
            print("⚠️  Cannot make prediction - missing feature data")

print("\n✅ Prediction complete!")
