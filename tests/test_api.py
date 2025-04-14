import requests

url = "http://localhost:8000/predict"
file_path = "data/daily.csv"

with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files, params={"date_from": "2021-06-27", "date_to": "2021-07-05"})

if response.status_code == 200:
    result = response.json()
    print(f"Best Model: {result['best_model']}")
    print(f"MAPE: {result['mape']:.4f}")
    print("First 5 Predictions:", result['predictions'][:5])
else:
    print(f"Error: {response.status_code}, {response.text}")