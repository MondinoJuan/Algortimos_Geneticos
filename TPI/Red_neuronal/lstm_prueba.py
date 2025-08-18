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

# Preparo los archivos necesarios para entrenar el modelo
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

# Debo limpiar las columnas con muchos datos faltantes porque interfieren mucho en el entreno de la RN.
# Elimino las columnas que tienen 85% o más datos faltantes
df_suelo_filtrado = df_suelo.dropna(thresh=len(df_suelo) * 0.15, axis=1)
df_clima_filtrado = df_clima.dropna(thresh=len(df_clima) * 0.15, axis=1)
df_semillas_filtrado = df_semillas.dropna(thresh=len(df_semillas) * 0.15, axis=1)


# Crear nueva columna 'departamento' indicando a que departamento corresponde cada dato de suelo
df_suelo_filtrado['departamento_nombre'] = df_suelo_filtrado.apply(
    lambda row: transformo_coord_a_depto([row['latitude'], row['longitude']]),
    axis=1
)
# Eliminar columnas latitude y longitude
df_suelo_filtrado = df_suelo_filtrado.drop(columns=['latitude', 'longitude'])
# Reordenar para que 'departamento' quede como primera columna
cols = ['departamento_nombre'] + [col for col in df_suelo_filtrado.columns if col != 'departamento_nombre']
df_suelo_filtrado = df_suelo_filtrado[cols]

# Calcular el promedio de las columnas numéricas para cada departamento
df_suelo_promedio = df_suelo_filtrado.groupby('departamento_nombre', as_index=False).mean(numeric_only=True)


# ---------------------------------------------
# UNIÓN DE LOS DATOS EN UN SOLO DATAFRAME
# Crear archivo con todos los datos necesarios, las columnas mostraran los datos de las semillas, una columna que tenga una matriz con los datos climásticos de los últimos
# n meses, o una columna por cada mes previo y en esa columna tener un array de los datos climáticos (o viceversa), y las columnas de los datos del suelo.
# ---------------------------------------------

# Combino los datos de las semillas con los datos del suelo por medio del departamento
df_combino_semillas_suelo = pd.merge(df_semillas_filtrado, df_suelo_promedio, on='departamento_nombre', how='inner')

# Ordeno por columna "anio"
df_combino_semillas_suelo = df_combino_semillas_suelo.sort_values(by='anio').reset_index(drop=True)


'''
A df_combino_semillas_suelo le agrego las columnas temperatura_media_C, humedad_relativa_%, velocidad_viento_m_s, velocidad_viento_km_h, 
precipitacion_mm_mes con los datos recuperados del archivo "clima_nasa_mensual_44_anios.csv, cada fila de las columnas tendrá un arreglo 
de tamaño n que tenga los datos correspondiente a esa columna de los ultimos n meses previos al año que se lee en la columna anio de 
df_combino_semillas_suelo
'''

def agregar_datos_climaticos1(df_combino, df_clima, n_meses):
    for mes in range(1, n_meses + 1):
        col_name = f'mes_{mes}'
        df_combino[col_name] = df_combino['anio'].apply(
            lambda x: df_clima[df_clima['fecha'].dt.year == x - 1][
                ['fecha', 'temperatura_media_C', 'humedad_relativa_%', 'velocidad_viento_m_s', 'velocidad_viento_km_h', 'precipitacion_mm_mes']
            ].tail(mes).values.tolist()
        )
    return df_combino

def agregar_datos_climaticos2(df_combino, df_clima, n_meses):
    df_clima = df_clima.sort_values('fecha')

    for variable in ['temperatura_media_C','humedad_relativa_%',
                     'velocidad_viento_m_s','velocidad_viento_km_h',
                     'precipitacion_mm_mes']:

        df_combino[variable] = df_combino['anio'].apply(
            lambda anio: df_clima[df_clima['fecha'].dt.year < anio][variable]
                          .tail(n_meses)
                          .tolist()
        )
    return df_combino

n_meses = 18

df_combino_semillas_suelo_clima = agregar_datos_climaticos1(df_combino_semillas_suelo, df_clima, n_meses)
df_combino_semillas_suelo_clima = agregar_datos_climaticos2(df_combino_semillas_suelo, df_clima, n_meses)

pd.to_csv(df_combino_semillas_suelo_clima, 'Red_neuronal/df_completo_para_RN_lstm.csv', index=False)



# Reordeno las columnas del archivo para que los datos de entrada queden al principio y los de salida (la produccion obtenida) al final
# Completar cuando se obtenga el archivo df_completo_para_RN_lstm.csv




# ---------------------------------------------
# ENTRENAMIENTO DEL MODELO
# Se utilizará el 80% de los datos para entrenar y el 20% para validar
# ---------------------------------------------
# Cargo el archivo con todos los datos necesarios
df_completo = pd.read_csv('Red_neuronal/df_completo_para_RN_lstm.csv')

# Selecciono las columnas que voy a usar como datos de entrada y salida

