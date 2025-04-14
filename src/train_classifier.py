import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import linregress, skew
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from collections import Counter
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)  # Specific to statsmodels warnings
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

# Load data with error handling
data_path = os.getenv("DATA_PATH", "./data/daily.csv")
try:
    df = pd.read_csv(data_path, parse_dates=["point_timestamp"], dayfirst=True)
    if "point_timestamp" not in df.columns or "point_value" not in df.columns:
        raise ValueError("CSV must contain 'point_timestamp' and 'point_value' columns")
    series = pd.Series(df["point_value"].values, index=df["point_timestamp"])
    series.index.freq = 'D'
    if len(series) < 28:  # Minimum for two windows
        raise ValueError("Insufficient data points (<28)")
except Exception as e:
    raise RuntimeError(f"Data loading failed: {e}")

# Enhanced feature extraction with validation
def extract_features(series_window):
    """Extract time series features with validation and NaN handling."""
    if not isinstance(series_window, pd.Series) or series_window.empty:
        return [0] * 7
    series_window = series_window.interpolate(method='linear').fillna(0)
    min_length = 14
    trend = linregress(range(len(series_window)), series_window.values)[0] if len(series_window) > 1 else 0
    seasonality = (seasonal_decompose(series_window, period=7, model='additive', extrapolate_trend='freq').seasonal.mean()
                   if len(series_window) >= min_length else
                   series_window.diff(7).mean() if len(series_window) >= 7 else 0)
    seasonality = 0 if np.isnan(seasonality) else seasonality
    acf_vals = acf(series_window, nlags=3)[1:2] if len(series_window) > 3 else [0]
    pacf_vals = pacf(series_window, nlags=3)[1:2] if len(series_window) > 3 else [0]
    rolling_mean = series_window.rolling(window=7, min_periods=1).mean().mean() if len(series_window) >= 1 else 0
    rolling_std = series_window.rolling(window=7, min_periods=1).std().mean() if len(series_window) >= 1 else 0
    skewness = skew(series_window.dropna().values) if len(series_window.dropna()) > 2 else 0
    features = [trend, seasonality, acf_vals[0], pacf_vals[0], rolling_mean, rolling_std, skewness]
    return [0 if np.isnan(f) or np.isinf(f) else f for f in features]

# Robust model evaluation
def evaluate_model(model_type, train, test):
    """Evaluate model with error handling and dynamic tuning."""
    if len(train) < 14 or len(test) < 1:
        return float("inf")
    try:
        if model_type == "Prophet" and len(train) < 20:
            return float("inf")
        elif model_type == "Prophet":
            df_prophet = pd.DataFrame({"ds": train.index, "y": train.values})
            model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_prophet)
            forecast = model.predict(pd.DataFrame({"ds": test.index}))["yhat"].values
        elif model_type == "ETS":
            model = ETSModel(train, trend="add", seasonal=None, freq="D")
            fit = model.fit(disp=False)  # Suppress convergence warnings
            forecast = fit.forecast(len(test)).values
        elif model_type == "ARIMA":
            model = ARIMA(train, order=(1, 1, 1))
            fit = model.fit()
            forecast = fit.forecast(len(test)).values
        elif model_type == "SARIMA":
            if len(train) < 7:
                return evaluate_model("ARIMA", train, test)
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            forecast = fit.forecast(len(test)).values
        forecast = np.nan_to_num(forecast, nan=0)
        mape = mean_absolute_percentage_error(test, forecast) if len(test) > 0 and not np.allclose(test, 0) else float("inf")
        return mape
    except Exception as e:
        print(f"Model {model_type} failed: {e}")
        return float("inf")

# Process windows with parallel execution
def process_window(series, window_size=30, step=10, min_window=14):
    X, y = [], []
    for ws in [window_size, min_window]:
        for i in range(0, len(series) - ws * 2, step):
            window = series[i:i + ws]
            test_window = series[i + ws:i + ws * 2]
            if len(window) < ws or len(test_window) < ws:
                break
            features = extract_features(window)
            mape_scores = {m: evaluate_model(m, window, test_window) for m in ["Prophet", "ETS", "ARIMA", "SARIMA"]}
            best_model = min(mape_scores, key=mape_scores.get, default="ARIMA")
            X.append(features)
            y.append(best_model)
    return np.array(X), np.array(y)

# Train classifier with enhanced logging
X, y = process_window(series)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Encoded labels: {y_encoded[:10]}... (total {len(y_encoded)} samples)")
print(f"LabelEncoder classes: {le.classes_}")
print(f"Label distribution: {Counter(le.inverse_transform(y_encoded))}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
smote = SMOTE(k_neighbors=4, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
print(f"Resampled label distribution: {Counter(y_resampled)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(classifier, X_resampled, y_resampled, cv=cv, n_jobs=-1)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

classifier.fit(X_resampled, y_resampled)
print(f"Classifier trained with shape X: {X_resampled.shape}, y: {y_resampled.shape}")

# Save models with validation
os.makedirs("models", exist_ok=True)
joblib.dump(classifier, "./models/classifier.pkl")
joblib.dump(le, "./models/label_encoder.pkl")
print("Models saved successfully. Verify at ./models/classifier.pkl and ./models/label_encoder.pkl")