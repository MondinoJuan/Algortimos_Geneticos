import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import obtener_datos_nasa_power, procesar_datos_mensuales


# -----------------------------
# CREACIÓN DEL MODELO LSTM
# -----------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Producción en kg
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model




# -----------------------------
# PROGRAMA
# -----------------------------

latitud_ejemplo = -31.4
longitud_ejemplo = -64.2
años_atras_ejemplo = 44

df_clima_diario = obtener_datos_nasa_power(latitud_ejemplo, longitud_ejemplo, años_atras_ejemplo)
if df_clima_diario.empty: #Si no hay datos climáticos para las coordenadas y años especificados terminar la ejecución
    raise ValueError("No se encontraron datos climáticos para las coordenadas y años especificados.")
df_clima = procesar_datos_mensuales(df_clima_diario)
# Recupero el menor año del que se tengan datos
df_clima['fecha'] = pd.to_datetime(df_clima['fecha'], errors='coerce')
min_year = df_clima['fecha'].min().year

# Recupero todos los datos de las semillas desde min_year en adelante
df_semillas = pd.read_csv('Recuperacion_de_datos/Semillas/Archivos generados/semillas_todas_concatenadas.csv')
df_semillas['fecha'] = pd.to_datetime(df_semillas['fecha'], errors='coerce')
df_semillas = df_semillas[df_semillas['fecha'].dt.year >= min_year + 1] # Aseguramos que las semillas tengan al menos un año de datos climáticos previos.

df_suelo = pd.read_csv('Recuperacion_de_datos/Suelos/suelo_unido.csv')


# Crear archivo con todos los datos necesarios, las columnas mostraran los datos de las semillas, una columna que tenga una matriz con los datos climásticos de los últimos
# n meses, o una columna por cada mes previo y en esa columna tener un array de los datos climáticos (o viceversa), y las columnas de los datos del suelo.


# Debo limpiar las columnas con muchos datos faltantes porque interfieren mucho en el entreno de la RN.