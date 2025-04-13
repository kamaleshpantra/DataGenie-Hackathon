from fastapi import FastAPI, UploadFile, HTTPException, Query
from io import BytesIO
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

app = FastAPI()

def extract_features(series):
    return [series.mean(), series.diff().dropna().mean() if len(series) > 1 else 0, series.var()]

def evaluate_model(model_type, train, test_or_future):
    try:
        if model_type == "Prophet":
            df = pd.DataFrame({"ds": train.index, "y": train.values})
            model = Prophet().fit(df)
            future = model.make_future_dataframe(periods=len(test_or_future), freq="D")
            forecast = model.predict(future)["yhat"][-len(test_or_future):].values
            mape = mean_absolute_percentage_error(test_or_future, forecast) if len(test_or_future) > 0 and not test_or_future.isna().all() else None
            return forecast.tolist(), mape
        elif model_type == "ETS":
            model = ETSModel(train, trend="add", seasonal=None, freq="D").fit()
            forecast = model.forecast(len(test_or_future)).values
            mape = mean_absolute_percentage_error(test_or_future, forecast) if len(test_or_future) > 0 and not test_or_future.isna().all() else None
            return forecast.tolist(), mape
        elif model_type == "ARIMA":
            model = ARIMA(train, order=(1, 1, 1)).fit()
            forecast = model.forecast(len(test_or_future)).values
            mape = mean_absolute_percentage_error(test_or_future, forecast) if len(test_or_future) > 0 and not test_or_future.isna().all() else None
            return forecast.tolist(), mape
        elif model_type == "SARIMA":
            if len(train) < 7: raise ValueError("Insufficient data for SARIMA.")
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()
            forecast = model.forecast(len(test_or_future)).values
            mape = mean_absolute_percentage_error(test_or_future, forecast) if len(test_or_future) > 0 and not test_or_future.isna().all() else None
            return forecast.tolist(), mape
        elif model_type == "LSTM":
            if len(train) < 20: raise ValueError("Insufficient data for LSTM.")
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
            X, y = [scaled_train[i-10:i, 0] for i in range(10, len(scaled_train))], [scaled_train[i, 0] for i in range(10, len(scaled_train))]
            X, y = np.array(X), np.array(y)
            X = torch.FloatTensor(X).unsqueeze(-1)
            y = torch.FloatTensor(y)
            model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
            model.load_state_dict(torch.load("./models/lstm_model.pth"))
            model.eval()
            scaled_test = scaler.transform(test_or_future.values.reshape(-1, 1))
            X_test = torch.FloatTensor([scaled_test[max(0, i-10):i, 0] for i in range(10, len(scaled_test) + 10)]).unsqueeze(-1)
            with torch.no_grad():
                forecast = model(X_test).numpy()
            forecast = scaler.inverse_transform(forecast)
            mape = mean_absolute_percentage_error(test_or_future[10:], forecast.flatten()) if len(test_or_future) > 10 and not test_or_future.isna().all() else None
            return forecast.tolist(), mape
    except Exception as e:
        print(f"Model error: {e}, falling back to ARIMA")
        model = ARIMA(train, order=(1, 1, 1)).fit()
        forecast = model.forecast(len(test_or_future)).values
        mape = mean_absolute_percentage_error(test_or_future, forecast) if len(test_or_future) > 0 and not test_or_future.isna().all() else None
        return forecast.tolist(), mape

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@app.post("/predict")
async def predict(file: UploadFile, date_from: str = Query(None), date_to: str = Query(None), period: int = Query(0)):
    if not file.filename.endswith(".csv"): raise HTTPException(status_code=400, detail="Please upload a CSV file")
    content = await file.read()
    data = pd.read_csv(BytesIO(content), parse_dates=["point_timestamp"], dayfirst=True)
    if len(data) < 5: raise ValueError("Dataset must have at least 5 rows")
    series = pd.Series(data["point_value"].values, index=data["point_timestamp"])
    series.index.freq = "D"
    if date_from and date_to:
        start, end = pd.to_datetime(date_from, dayfirst=True), pd.to_datetime(date_to, dayfirst=True)
        series = series[(series.index >= start) & (series.index <= end)]
    if len(series) < 2: raise ValueError("Insufficient data. Need at least 2 rows.")
    train_size = max(int(len(series) * 0.8), len(series) - 1)
    train, test = series[:train_size], series[train_size:]
    classifier = joblib.load("./models/classifier.pkl")
    features = extract_features(series)
    best_model = classifier.predict([features])[0]
    if period > 0:
        future_dates = pd.date_range(start=series.index[-1], periods=period + 1, freq="D")[1:]
        future_series = pd.Series(0, index=future_dates)
        forecast, mape = evaluate_model(best_model, train, future_series)
        mape = None
    else:
        if len(test) == 0: raise ValueError("Test set is empty.")
        forecast, mape = evaluate_model(best_model, train, test[:len(forecast)])
    return {
        "best_model": best_model,
        "mape": mape,
        "predictions": [{"point_timestamp": str(date), "predicted": pred} for date, pred in zip(test.index[:len(forecast)] if period == 0 else future_dates, forecast)]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)