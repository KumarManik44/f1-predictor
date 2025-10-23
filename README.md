# 🏎️ Formula 1 Race Winner Prediction System

An end-to-end machine learning system that predicts F1 Grand Prix winners and podium finishes using historical race data, qualifying results, and driver performance metrics.

## 🎯 Project Overview

This project demonstrates:
- ✅ **Data Engineering**: Collected 2,500+ race records from FastF1 and Jolpica F1 APIs (2020-2025)
- ✅ **Feature Engineering**: Created 60+ predictive features including driver form, grid position, and team performance
- ✅ **Machine Learning**: Trained XGBoost classifier achieving **91.56% accuracy** on 2025 season predictions
- ✅ **Model Deployment**: Built interactive Streamlit web app for real-time race predictions
- ✅ **Time-Series Analysis**: Implemented proper temporal cross-validation to prevent data leakage

## 📊 Model Performance

**Test Set (2025 Season - 19 Races):**
- **Overall Accuracy**: 91.56%
- **Podium Precision**: 71%
- **Podium Recall**: 74%
- **True Positives**: 42 correctly predicted podium finishes
- **False Positives**: 17
- **False Negatives**: 15

## 🔑 Key Features

### Data Collection
- Historical race results (2020-2025)
- Qualifying session data
- Driver and constructor standings
- Practice session performance (optional)

### Engineered Features
1. **Grid Position** (67.4% feature importance) - Starting position from qualifying
2. **Driver Form** - Average points from last 5 races
3. **Recent Performance** - Average finishing position (last 5 races)
4. **Constructor Season Points** - Team cumulative points
5. **Driver Season Points** - Driver championship position
6. **Circuit-Specific Stats** (future enhancement)

## 🚀 Installation & Setup

### Prerequisites
Install Homebrew (Mac only)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Install OpenMP (required for XGBoost)
brew install libomp

### Install Dependencies
pip install fastf1 pandas numpy scikit-learn xgboost streamlit

### Configure FastF1 API
import fastf1
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
fastf1.Cache.enable_cache('f1_cache')


## 📁 Project Structure

f1-predictor/
├── data/
│ ├── race_results_2020_2024.csv
│ ├── race_results_2025.csv
│ ├── qualifying_results_2020_2024.csv
│ ├── qualifying_results_2025.csv
│ └── f1_dataset_with_features.csv
├── api_verify.py # API connection test
├── collect_data.py # Historical race data collection
├── collect_qualifying.py # Qualifying data collection
├── collect_2025_data.py # Current season data
├── merge_explore_data.py # Data preprocessing
├── feature_engineering.py # Feature creation
├── train_model.py # Model training & evaluation
├── predict_next_race.py # Race winner prediction
├── streamlit_app.py # Interactive web application
└── README.md


### Training Strategy
- **Training Data**: 2020-2024 seasons (2,118 records)
- **Test Data**: 2025 season (379 records)
- **Validation**: Time-based split (no data leakage)
- **Target Variable**: Binary classification (Podium finish: Top 3 vs. No podium)

### Feature Importance
| Feature | Importance |
|---------|-----------|
| Grid Position | 67.4% |
| Driver Form (Last 5 races) | 10.0% |
| Constructor Season Points | 8.9% |
| Driver Season Points | 7.0% |
| Avg Position (Last 5 races) | 6.7% |

## 🎓 Key Learnings & Challenges

### Challenges Faced
1. **API Deprecation**: Original Ergast API shut down → Migrated to Jolpica F1 API
2. **Rate Limiting**: Implemented delays (1-5 seconds) to avoid API throttling
3. **Feature Leakage**: Used time-based validation to prevent future data contamination
4. **Class Imbalance**: Only 15% of finishes are podiums → Focused on precision/recall metrics
5. **Missing Data**: Handled incomplete qualifying/race records gracefully

### Technical Skills Demonstrated
- **API Integration**: FastF1, Jolpica F1 (RESTful APIs)
- **Data Engineering**: ETL pipelines, feature engineering, temporal data handling
- **Machine Learning**: XGBoost, scikit-learn, classification metrics
- **Model Evaluation**: Confusion matrix, precision/recall, feature importance
- **Deployment**: Streamlit web application
- **Version Control**: Git/GitHub workflow

## 🔮 Future Enhancements

1. **Advanced Features**:
   - Weather data (rainfall, track temperature)
   - Practice session lap times (FP1, FP2, FP3)
   - Tire strategy predictions
   - Safety car probability
   - Driver vs. teammate head-to-head stats

2. **Model Improvements**:
   - LSTM/GRU for sequential race-to-race dependencies
   - Ensemble methods (XGBoost + Random Forest + Neural Network)
   - Monte Carlo simulation for race outcome probabilities
   - Circuit-specific models (street circuits vs. traditional tracks)

3. **Deployment**:
   - FastAPI backend for REST API access
   - Docker containerization
   - Cloud deployment (AWS/GCP)
   - Real-time predictions during race weekends

## 🛠️ Technologies Used

- **Python 3.13**
- **FastF1** - F1 data acquisition
- **Pandas** - Data manipulation
- **XGBoost** - Machine learning model
- **Scikit-learn** - Model evaluation
- **Streamlit** - Web application framework
- **NumPy** - Numerical computing

## 📄 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ for F1 fans and ML enthusiasts**
