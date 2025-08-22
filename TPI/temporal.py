import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from pathlib import Path
import sys, os
from shapely.geometry import MultiPoint
import numpy as np

# Agrego la carpeta raíz del proyecto (TPI) al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import obtener_datos_nasa_power, procesar_datos_mensuales
from Mapa.GIS.departamento import encontrar_departamento as transformo_coord_a_depto


df_final = pd.read_csv('df_semillas_suelo_clima.csv')
column_order = [
    "cultivo_nombre", "anio", "departamento_nombre", "organic_carbon", "ph", "clay", "silt", "sand", 
    "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "velocidad_viento_km_h", 
    "precipitacion_mm_mes", "superficie_sembrada_ha",   # Entradas
    "rendimiento_kgxha"                                 # Salida esperada
]
df_final = df_final[column_order]

df_final.to_csv('df_semillas_suelo_clima.csv', index=False)