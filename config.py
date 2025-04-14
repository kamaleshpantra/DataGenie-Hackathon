import os

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "data/daily.csv")
    MODEL_DIR = os.getenv("MODEL_DIR", "models")