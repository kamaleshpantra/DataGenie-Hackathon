import requests
with open('./data/daily.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files, params={'period': 0})
    print(response.json())