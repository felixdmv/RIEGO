import os
import requests
import pandas as pd
from datetime import datetime

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmZWxpeG1pdmkyMDAyQGdtYWlsLmNvbSIsImp0aSI6ImMyMDIyY2U5LWZjNWItNGM3NC05MGRiLTk5MTcyZDBjNjFiYiIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzcwOTg4NDEyLCJ1c2VySWQiOiJjMjAyMmNlOS1mYzViLTRjNzQtOTBkYi05OTE3MmQwYzYxYmIiLCJyb2xlIjoiIn0.3SYgPoIKWo8zDN-jjHIKfZ9FbvXeXLkqhskHyYeBoHg"
ID_ESTACION = "2172Y"

url = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2026-06-23T00:00:00UTC/fechafin/2026-06-26T23:59:59UTC/estacion/{ID_ESTACION}"
headers = {
    'cache-control': "no-cache",
    'api_key': API_KEY
}

print("Requesting AEMET data...")
response = requests.get(url, headers=headers)
print("Response code:", response.status_code)
json_data = response.json()
print("Initial response JSON:", json_data)
if json_data.get("estado") == 200:
    data_url = json_data["datos"]
    resp_data = requests.get(data_url, headers=headers)
    df = pd.DataFrame(resp_data.json())
    print("\nDownloaded data:")
    print(df.to_string())
else:
    print("Failed to download data.")
