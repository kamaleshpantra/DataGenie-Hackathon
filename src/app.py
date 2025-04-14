from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
from src.predictions import generate_predictions
from fastapi.encoders import jsonable_encoder
import io
from datetime import datetime

app = FastAPI(title="DataGenie Time Series API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_dates(date_from: str, date_to: str) -> tuple:
    try:
        d_from = datetime.strptime(date_from, "%Y-%m-%d")
        d_to = datetime.strptime(date_to, "%Y-%m-%d")
        if d_from > d_to:
            raise ValueError("date_from must be less than or equal to date_to")
        return d_from, d_to
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")

@app.post("/predict")
async def predict(date_from: str, date_to: str, file: UploadFile = File(...)):
    try:
        d_from, d_to = validate_dates(date_from, date_to)
        logger.info(f"Received request for date range {date_from} to {date_to} with file {file.filename}")

        contents = await file.read()
        if not file.filename.lower().endswith('.csv'):
            raise ValueError("Unsupported file type. Please upload a .csv file.")
        df = pd.read_csv(io.BytesIO(contents), parse_dates=["point_timestamp"], dayfirst=True, on_bad_lines='skip')
        if "point_timestamp" not in df.columns or "point_value" not in df.columns:
            raise ValueError("CSV must contain 'point_timestamp' and 'point_value' columns")
        series = pd.Series(df["point_value"].values, index=df["point_timestamp"])
        series.index.freq = 'D'
        if len(series) == 0:
            raise ValueError("No valid data points in CSV")
        logger.info(f"Processed {len(series)} points from {file.filename}")

        result = generate_predictions(series, date_from, date_to)
        response = {
            "forecastability_score": result["forecastability_score"],
            "number_of_batch_fits": result["number_of_batch_fits"],
            "mape": result["mape"],
            "avg_time_taken_per_fit_in_seconds": result["avg_time_taken_per_fit_in_seconds"],
            "results": result["results"]
        }
        logger.info(f"Returning prediction response with {len(result['results'])} results")
        return JSONResponse(content=jsonable_encoder(response))
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)