import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from statsmodels.tsa.stattools import acf
from scipy.stats import linregress, entropy
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBClassifier
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

def extract_features(series):
    """Extract statistical and temporal features from a time series."""
    series = series.interpolate(method='linear')
    trend = linregress(range(len(series)), series.values)[0] if len(series) > 1 else 0
    seasonality = seasonal_decompose(series, period=7, model='additive', extrapolate_trend='freq').seasonal.mean() if len(series) >= 7 else 0
    acf_vals = acf(series, nlags=3)[1:2] if len(series) > 3 else [0]  # Only lag 1
    return [trend, seasonality, acf_vals[0]]

def evaluate_model(model_type, train, test, window_idx):
    """Evaluate a time series model and return MAPE."""
    try:
        if model_type == "Prophet":
            df = pd.DataFrame({"ds": train.index, "y": train.values})
            model = Prophet().fit(df)
            forecast = model.predict(pd.DataFrame({"ds": test.index}))["yhat"].values
            return mean_absolute_percentage_error(test, forecast) if len(test) > 0 else float("inf")
        elif model_type == "ETS":
            model = ETSModel(train, trend="add", seasonal=None, freq="D").fit()
            forecast = model.forecast(len(test)).values
            return mean_absolute_percentage_error(test, forecast) if len(test) > 0 else float("inf")
        elif model_type == "ARIMA":
            model = ARIMA(train, order=(1, 1, 1)).fit()
            forecast = model.forecast(len(test)).values
            return mean_absolute_percentage_error(test, forecast) if len(test) > 0 else float("inf")
        elif model_type == "SARIMA":
            if len(train) < 7: return evaluate_model("ARIMA", train, test, window_idx)
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False).fit()
            forecast = model.forecast(len(test)).values
            return mean_absolute_percentage_error(test, forecast) if len(test) > 0 else float("inf")
        elif model_type == "LSTM":
            if len(train) < 30: return evaluate_model("ARIMA", train, test, window_idx)
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
            if len(scaled_train) < 30:
                return float("inf")
            X, y = [scaled_train[i-30:i, 0] for i in range(30, len(scaled_train))], [scaled_train[i, 0] for i in range(30, len(scaled_train))]
            X, y = np.array(X), np.array(y)
            if len(X) == 0:
                return float("inf")
            X = torch.FloatTensor(X).unsqueeze(-1)
            y = torch.FloatTensor(y)
            model = LSTMModel(input_size=1, hidden_size=150, output_size=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            batch_size = 32
            for epoch in range(20):
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size] if i+batch_size <= len(X) else X[i:]
                    batch_y = y[i:i+batch_size] if i+batch_size <= len(y) else y[i:]
                    if len(batch_X) == 0 or len(batch_y) == 0:
                        continue
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            scaled_test = scaler.transform(test.values.reshape(-1, 1))
            X_test = torch.FloatTensor([scaled_test[max(0, i-30):i, 0] for i in range(30, len(scaled_test) + 30)]).unsqueeze(-1)
            with torch.no_grad():
                forecast = model(X_test).numpy()
            forecast = scaler.inverse_transform(forecast)
            lstm_mape = mean_absolute_percentage_error(test[30:], forecast.flatten()) if len(test) > 30 else float("inf")
            print(f"LSTM MAPE for window {window_idx}: {lstm_mape:.4f}")
            torch.save(model.state_dict(), "../models/lstm_model.pth")
            return lstm_mape
    except Exception as e:
        print(f"Error in {model_type} for window {window_idx}: {e}")
        return float("inf")

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_classifier():
    """Train a classifier to predict the best time series model."""
    df = pd.read_csv("./data/daily.csv", parse_dates=["point_timestamp"], dayfirst=True)
    series = pd.Series(df["point_value"].values, index=df["point_timestamp"])
    series.index.freq = 'D'
    if len(series) < 30: raise ValueError("Dataset must have at least 30 rows for LSTM")
    if series.isnull().any(): raise ValueError("Dataset contains missing data")
    window_size = 30
    X, y = [], []
    def process_window(i):
        window = series[i-window_size:i]
        test_window = series[i:i+window_size]
        features = extract_features(window)
        mape_scores = {m: evaluate_model(m, window, test_window, i) for m in ["Prophet", "ETS", "ARIMA", "SARIMA", "LSTM"]}
        best_model = min(mape_scores.items(), key=lambda x: x[1])[0]
        print(f"Window {i}: Best Model {best_model}, MAPE {min(mape_scores.values()):.4f}")
        return features, best_model
    results = Parallel(n_jobs=-1)(delayed(process_window)(i) for i in range(window_size, len(series) - window_size, 10))
    for features, best_model in results:
        X.append(features)
        y.append(best_model)
    if not X:
        raise ValueError("Insufficient data for multiple windows. Need at least 30 rows.")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    classifier = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, np.array(X), y_encoded, cv=cv)
    print(f"Classifier Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    classifier.fit(np.array(X), y_encoded)  # Fit on full data to get feature importances
    importances = classifier.feature_importances_
    print("Feature Importances:", importances)
    joblib.dump(classifier, "./models/classifier.pkl")
    joblib.dump(le, "./models/label_encoder.pkl")

if __name__ == "__main__":
    train_classifier()