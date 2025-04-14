import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import linregress, skew, zscore
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import logging
import warnings
import time
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

classifier = joblib.load("models/classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def convert_to_native_types(data):
    """Convert NumPy types to native Python types for JSON compatibility."""
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

def extract_features(series_window):
    """Extract time series features with consistent NaN handling."""
    if not isinstance(series_window, pd.Series) or series_window.empty:
        return [0] * 7
    series_window = series_window.interpolate(method='linear').fillna(0)
    trend = linregress(range(len(series_window)), series_window.values)[0] if len(series_window) > 1 else 0
    seasonality = (seasonal_decompose(series_window, period=7, model='additive', extrapolate_trend='freq').seasonal.mean()
                   if len(series_window) >= 14 else 0)
    seasonality = 0 if np.isnan(seasonality) else seasonality
    acf_vals = acf(series_window, nlags=3)[1:2] if len(series_window) > 3 else [0]
    pacf_vals = pacf(series_window, nlags=3)[1:2] if len(series_window) > 3 else [0]
    rolling_mean = series_window.rolling(window=7, min_periods=1).mean().mean() if len(series_window) >= 1 else 0
    rolling_std = series_window.rolling(window=7, min_periods=1).std().mean() if len(series_window) >= 1 else 0
    skewness = skew(series_window.dropna().values) if len(series_window.dropna()) > 2 else 0
    return [trend, seasonality, acf_vals[0], pacf_vals[0], rolling_mean, rolling_std, skewness]

def evaluate_and_predict(model_type: str, train: pd.Series, test: pd.Series) -> tuple:
    """Evaluate model and generate predictions with error handling."""
    if len(train) < 14:
        return np.zeros(len(test)), float("inf"), 0
    try:
        start_time = time.time()
        if model_type == "Prophet" and len(train) < 20:
            return np.zeros(len(test)), float("inf"), 0
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
                return evaluate_and_predict("ARIMA", train, test)
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            forecast = fit.forecast(len(test)).values
        fit_time = time.time() - start_time
        forecast = np.nan_to_num(forecast, nan=0)
        mape = mean_absolute_percentage_error(test, forecast) if len(test) > 0 and not np.allclose(test, 0) else float("inf")
        return forecast, mape, fit_time
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        return np.zeros(len(test)), float("inf"), 0

def detect_anomaly(actual: Optional[float], predicted: float, threshold: float = 2.0) -> str:
    """Detect anomalies using z-score."""
    actual_val = actual if actual is not None and not np.isnan(actual) else 0
    difference = actual_val - predicted
    z_score = zscore([difference])[0] if difference != 0 else 0
    return "yes" if abs(z_score) > threshold else "no"

def generate_predictions(series: pd.Series, date_from: str, date_to: str, window_size: int = 30, num_batches: int = 5) -> Dict:
    """Generate time series predictions with the best model."""
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series with a datetime index")
    if len(series) < window_size:
        raise ValueError(f"Time series length ({len(series)}) must be at least {window_size}")

    initial_window = series[:window_size]
    features = extract_features(initial_window)
    predicted_model_idx = classifier.predict([features])[0]
    best_model = label_encoder.inverse_transform([predicted_model_idx])[0]
    logger.info(f"Predicted best model: {best_model}")

    train = series[series.index < pd.to_datetime(date_from)]
    test_dates = pd.date_range(start=date_from, end=date_to, freq='D')
    test = series[series.index.isin(test_dates)] if not test_dates.empty else pd.Series(index=test_dates, dtype=float)
    logger.info(f"Train data length: {len(train)}, Test data length: {len(test)}")

    if len(test) == 0:
        raise ValueError("No test data available for the specified date range")

    fit_times = []
    forecasts = []
    for _ in range(num_batches):
        forecast, mape, fit_time = evaluate_and_predict(best_model, train, test)
        fit_times.append(fit_time)
        forecasts.append(forecast)
    avg_fit_time = np.mean(fit_times) if fit_times else 0
    final_forecast = np.mean(forecasts, axis=0) if forecasts else np.zeros(len(test))

    forecastability_score = min(10, 1 / mape) if mape > 0 and mape < float("inf") else 0

    predictions = []
    for i, (ts, pred) in enumerate(zip(test_dates, final_forecast)):
        actual = test.iloc[i] if i < len(test) else None
        is_anomaly = detect_anomaly(actual, pred)
        predictions.append({
            "timestamp": str(ts),
            "point_value": actual,
            "predicted": pred,
            "is_anomaly": is_anomaly
        })

    result = {
        "forecastability_score": forecastability_score,
        "number_of_batch_fits": num_batches,
        "mape": mape if mape < float("inf") else None,
        "avg_time_taken_per_fit_in_seconds": avg_fit_time,
        "results": predictions
    }
    return convert_to_native_types(result)

try:
    df = pd.read_csv("data/daily.csv", parse_dates=["point_timestamp"], dayfirst=True)
    if "point_timestamp" not in df.columns or "point_value" not in df.columns:
        raise ValueError("CSV must contain 'point_timestamp' and 'point_value' columns")
    series = pd.Series(df["point_value"].values, index=df["point_timestamp"])
    series.index.freq = 'D'
except FileNotFoundError:
    logger.error("data/daily.csv not found")
    raise
except Exception as e:
    logger.error(f"Data loading error: {e}")
    raise

results = generate_predictions(series, "2021-06-27", "2021-07-05")
print(f"Forecastability Score: {results['forecastability_score']}")
print(f"MAPE: {results['mape']}")
print(f"First 5 predictions: {results['results'][:5]}")