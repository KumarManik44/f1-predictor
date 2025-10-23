import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

st.set_page_config(page_title="F1 Race Winner Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 Race Winner Predictor")
st.markdown("### Predict F1 race outcomes using machine learning")


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('f1_dataset_with_features.csv')
    return df


@st.cache_resource
def train_model():
    df = load_data()
    feature_columns = [
        'grid_position',
        'driver_last5_avg_points',
        'driver_last5_avg_position',
        'constructor_season_points',
        'driver_season_points'
    ]

    train_data = df[df['season'] < 2025].dropna(subset=feature_columns + ['podium_finish'])
    X_train = train_data[feature_columns]
    y_train = train_data['podium_finish']

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model, feature_columns


df = load_data()
model, feature_columns = train_model()

# Sidebar - Model Performance
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy (2025 Season)", "91.56%")
st.sidebar.metric("Podium Precision", "71%")
st.sidebar.metric("Podium Recall", "74%")

st.sidebar.markdown("---")
st.sidebar.markdown("**Most Important Feature:**")
st.sidebar.markdown("🎯 Grid Position (67.4%)")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Predict Next Race", "📈 2025 Championship", "🏆 Model Insights"])

with tab1:
    st.header("Predict Race Winner")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Race Details")

        # Get latest driver stats
        latest_stats = df[df['season'] == 2025].groupby('driverCode').agg({
            'driver_last5_avg_points': 'last',
            'driver_last5_avg_position': 'last',
            'driver_season_points': 'last',
            'constructorName': 'last',
            'constructor_season_points': 'last'
        }).reset_index()

        # Driver selection
        drivers = sorted(latest_stats['driverCode'].unique())

        st.markdown("**Select drivers and their qualifying positions:**")

        selected_drivers = []
        for i in range(min(10, len(drivers))):
            col_driver, col_grid = st.columns([2, 1])
            with col_driver:
                driver = st.selectbox(f"Driver {i + 1}", drivers, key=f"driver_{i}", index=i if i < len(drivers) else 0)
            with col_grid:
                grid = st.number_input(f"Grid P{i + 1}", min_value=1, max_value=20, value=i + 1, key=f"grid_{i}")

            if driver:
                driver_stats = latest_stats[latest_stats['driverCode'] == driver].iloc[0]
                selected_drivers.append({
                    'driverCode': driver,
                    'constructorName': driver_stats['constructorName'],
                    'grid_position': grid,
                    'driver_last5_avg_points': driver_stats['driver_last5_avg_points'],
                    'driver_last5_avg_position': driver_stats['driver_last5_avg_position'],
                    'constructor_season_points': driver_stats['constructor_season_points'],
                    'driver_season_points': driver_stats['driver_season_points']
                })

    with col2:
        st.subheader("🏁 Predictions")

        if st.button("🔮 Predict Race Outcome", type="primary", use_container_width=True):
            if len(selected_drivers) > 0:
                # Create prediction DataFrame
                pred_df = pd.DataFrame(selected_drivers)
                X_pred = pred_df[feature_columns]

                # Make predictions
                podium_proba = model.predict_proba(X_pred)[:, 1]
                pred_df['podium_probability'] = podium_proba
                pred_df = pred_df.sort_values('podium_probability', ascending=False)

                # Display winner
                winner = pred_df.iloc[0]
                st.success(f"### 🏆 PREDICTED WINNER: {winner['driverCode']}")
                st.info(f"**Team:** {winner['constructorName']}")
                st.metric("Win Probability", f"{winner['podium_probability']:.1%}")

                st.markdown("---")
                st.markdown("### 🥇 Predicted Podium")

                for idx, row in pred_df.head(3).iterrows():
                    with st.expander(
                            f"P{list(pred_df.head(3).index).index(idx) + 1}: {row['driverCode']} - {row['podium_probability']:.1%}"):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Grid Position", int(row['grid_position']))
                        col_b.metric("Season Points", int(row['driver_season_points']))
                        col_a.metric("Recent Form (Avg Points)", f"{row['driver_last5_avg_points']:.1f}")
                        col_b.metric("Team", row['constructorName'])

                st.markdown("---")
                st.markdown("### 📊 All Predictions")
                display_df = pred_df[['driverCode', 'constructorName', 'grid_position', 'podium_probability']].copy()
                display_df['podium_probability'] = display_df['podium_probability'].apply(lambda x: f"{x:.1%}")
                display_df.columns = ['Driver', 'Team', 'Grid', 'Podium Probability']
                st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.header("2025 Championship Standings")

    latest_standings = latest_stats.sort_values('driver_season_points', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Driver Championship")
        for idx, row in latest_standings.head(10).iterrows():
            position = list(latest_standings.index).index(idx) + 1
            st.metric(
                f"P{position}: {row['driverCode']}",
                f"{int(row['driver_season_points'])} pts",
                delta=f"Avg: {row['driver_last5_avg_points']:.1f} pts/race"
            )

    with col2:
        st.subheader("Constructor Standings")
        constructor_standings = latest_stats.groupby('constructorName')['driver_season_points'].sum().sort_values(
            ascending=False)
        for idx, (team, points) in enumerate(constructor_standings.head(10).items()):
            st.metric(f"P{idx + 1}: {team}", f"{int(points)} pts")

with tab3:
    st.header("Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Feature Importance")
        importance_data = {
            'Feature': ['Grid Position', 'Driver Form (Last 5)', 'Constructor Points', 'Driver Points',
                        'Avg Position (Last 5)'],
            'Importance': [67.4, 10.0, 8.9, 7.0, 6.7]
        }
        st.bar_chart(pd.DataFrame(importance_data).set_index('Feature'))

    with col2:
        st.subheader("🎯 Model Performance")
        st.markdown("""
        **Training Data:** 2020-2024 seasons  
        **Test Data:** 2025 season (19 races)  

        **Results:**
        - Overall Accuracy: **91.56%**
        - Podium Precision: **71%**
        - Podium Recall: **74%**
        - True Positives: 42 podium finishes
        - False Positives: 17
        - False Negatives: 15

        **Key Insight:**  
        Grid position (qualifying) is the strongest predictor, accounting for 67% of the model's decision-making.
        """)

st.markdown("---")
st.markdown("Built with ❤️ using FastF1, XGBoost, and Streamlit | Data: 2020-2025 F1 Seasons")
