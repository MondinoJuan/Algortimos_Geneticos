import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from pathlib import Path
import sys, os
from shapely.geometry import MultiPoint
import numpy as np


df_final = pd.read_csv("df_semillas_suelo_clima.csv")

cols_cosecha = [
    "cultivo_nombre", "anio", "departamento_nombre", "organic_carbon", "ph", "clay", "silt", "sand", 
    "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "velocidad_viento_km_h", 
    "precipitacion_mm_mes", "superficie_sembrada_ha",                               # Entradas
    "superficie_cosechada_ha"                 # Salida esperada
]

cols_prod = [
    "cultivo_nombre", "anio", "departamento_nombre", "organic_carbon", "ph", "clay", "silt", "sand", 
    "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "velocidad_viento_km_h", 
    "precipitacion_mm_mes", "superficie_sembrada_ha",                               # Entradas
    "produccion_tn"                 # Salida esperada
]

cols_rend = [
    "cultivo_nombre", "anio", "departamento_nombre", "organic_carbon", "ph", "clay", "silt", "sand", 
    "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "velocidad_viento_km_h", 
    "precipitacion_mm_mes", "superficie_sembrada_ha",                               # Entradas
    "rendimiento_kgxha"                 # Salida esperada
]

df_con_cosecha = df_final[cols_cosecha]
df_con_prod = df_final[cols_prod]
df_con_rend = df_final[cols_rend]

df_con_cosecha.to_csv('df_con_cosecha.csv', index=False)
df_con_prod.to_csv('df_con_prod.csv', index=False)
df_con_rend.to_csv('df_con_rend.csv', index=False)

