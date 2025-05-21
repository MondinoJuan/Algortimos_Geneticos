import requests

API_KEY = "tu_clave_ign"
lat = 48.8584
lon = 2.2945

url = f"https://wxs.ign.fr/{API_KEY}/alti/rest/elevation.json"
params = {
    "lat": lat,
    "lon": lon
}

response = requests.get(url, params=params)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code)