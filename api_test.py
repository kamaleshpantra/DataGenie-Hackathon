import requests

url = "http://localhost:5000/predict"
files = {'file': open('data/daily.csv', 'rb')}
data = {'date_from': '27-06-2021', 'date_to': '05-07-2021'}

response = requests.post(url, files=files, data=data)
if response.status_code == 200:
    result = response.json()
    print(len(result['predictions']))  # Prints 9 if successful
else:
    print(f"Error: {response.status_code} - {response.text}")