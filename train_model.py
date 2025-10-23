import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

print("Loading dataset with features...\n")

# Load data
df = pd.read_csv('f1_dataset_with_features.csv')

print(f"✅ Loaded {len(df)} records\n")

# Select features for training
feature_columns = [
    'grid_position',
    'driver_last5_avg_points',
    'driver_last5_avg_position',
    'constructor_season_points',
    'driver_season_points'
]

target = 'podium_finish'  # Predicting top-3 finish

# Remove rows with missing values in features
df_clean = df.dropna(subset=feature_columns + [target])
print(f"✅ Clean dataset: {len(df_clean)} records (removed {len(df) - len(df_clean)} with missing values)\n")

# Prepare features and target
X = df_clean[feature_columns]
y = df_clean[target]

# Time-based split: Train on 2020-2024, Test on 2025
print("📊 Creating time-based train/test split...")
train_data = df_clean[df_clean['season'] < 2025]
test_data = df_clean[df_clean['season'] == 2025]

X_train = train_data[feature_columns]
y_train = train_data[target]
X_test = test_data[feature_columns]
y_test = test_data[target]

print(f"   - Training set: {len(X_train)} records (2020-2024)")
print(f"   - Test set: {len(X_test)} records (2025)")
print(f"   - Podium rate in training: {y_train.mean():.1%}")
print(f"   - Podium rate in test: {y_test.mean():.1%}\n")

# Train XGBoost model
print("🤖 Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ Model training complete!\n")

# Make predictions
print("🔮 Making predictions on test set (2025 season)...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of podium

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Performance on 2025 Season:")
print(f"   - Accuracy: {accuracy:.2%}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Podium', 'Podium']))

# Confusion Matrix
print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")

# Feature Importance
print("\n🎯 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Save model predictions
test_data_with_pred = test_data.copy()
test_data_with_pred['predicted_podium'] = y_pred
test_data_with_pred['podium_probability'] = y_pred_proba

output_file = 'model_predictions_2025.csv'
test_data_with_pred.to_csv(output_file, index=False)
print(f"\n💾 Predictions saved to: {output_file}")

# Show sample predictions
print("\n🏁 Sample Predictions (2025 races):")
display_cols = ['season', 'round', 'driverCode', 'grid_position', 'podium_probability',
                'predicted_podium', 'podium_finish', 'position']
print(test_data_with_pred[display_cols].head(30))
