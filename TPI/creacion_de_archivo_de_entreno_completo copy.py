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
# Filtro el departamento
# -----------------------------
def solo_departamento(coord):
    depto, _ = transformo_coord_a_depto(coord)
    return depto

# Calculo el centroide
def calcular_centroide(coord_list):
    multipoint = MultiPoint(coord_list)
    centroid = multipoint.centroid
    return (centroid.x, centroid.y)  # devuelve (lon, lat)


def agregar_clima_por_departamento(df_semillas, df_suelo_promedio, años_atras_ejemplo, n_meses=18):
    # Copia para no modificar el original
    df_semillas = df_semillas.copy()
    
    # CORREGIR: Asegurar que 'anio' sea solo el año (entero)
    '''if df_semillas['anio'].dtype == 'object':
        # Si es string con datetime, extraer solo el año
        df_semillas['anio'] = pd.to_datetime(df_semillas['anio'], errors='coerce').dt.year
    elif 'datetime' in str(df_semillas['anio'].dtype):
        df_semillas['anio'] = df_semillas['anio'].dt.year'''

    #print(f"AAAAAAAAAAAAAAAAAAAAAAA --- Tipos de datos después de corrección - anio: {df_semillas['anio'].dtype}")
    #print(f"Ejemplo de años: {df_semillas['anio'].head().tolist()}")

    # Iterar sobre cada departamento en df_suelo_promedio
    for _, row in df_suelo_promedio.iterrows():
        depto = row['departamento_nombre']
        lon_str, lat_str = row['coords'].strip("()").split(",")
        lon, lat = float(lon_str), float(lat_str)

        #print(f"Procesando departamento: {depto}")

        # Obtener clima para este centroide
        df_clima_diario = obtener_datos_nasa_power(lat, lon, años_atras_ejemplo)
        if df_clima_diario.empty:
            print(f"⚠️ No se encontraron datos climáticos para {depto}")
            continue

        df_clima = procesar_datos_mensuales(df_clima_diario)
        df_clima['fecha'] = pd.to_datetime(df_clima['fecha'], errors='coerce')
        df_clima_filtrado = df_clima.dropna(thresh=len(df_clima) * 0.15, axis=1).copy()
        #print(f"AAAAAAAAAAAAAAAAAAAAAAA --- El tipo de dato de fecha es: {df_clima_filtrado['fecha'].dtype}")

        #print(f"Rango de fechas clima: {df_clima_filtrado['fecha'].min()} a {df_clima_filtrado['fecha'].max()}")

        # Filtrar semillas de este departamento
        mask = df_semillas['departamento_nombre'] == depto
        #print(f"Registros de semillas en {depto}: {mask.sum()}")

        for variable in ['temperatura_media_C','humedad_relativa_%',
                 'velocidad_viento_m_s','velocidad_viento_km_h',
                 'precipitacion_mm_mes']:
            
            #df_semillas[variable] = df_semillas[variable].astype(object)

            def obtener_clima_previo(año_siembra):
                """
                Obtiene los últimos n_meses de datos climáticos anteriores al año de siembra
                """
                # Crear fecha límite: 31 de diciembre del año anterior
                fecha_limite = pd.Timestamp(year=año_siembra-1, month=12, day=31)
                
                # Filtrar datos anteriores a la fecha límite
                clima_previo = df_clima_filtrado[df_clima_filtrado['fecha'] <= fecha_limite]
                
                if len(clima_previo) == 0:
                    print(f"⚠️ No hay datos climáticos para {depto} antes de {año_siembra}")
                    return []
                
                # Tomar los últimos n_meses y retornar como lista
                valores = clima_previo[variable].tail(n_meses).tolist()
                
                # Si no tenemos suficientes meses, rellenar con NaN
                while len(valores) < n_meses:
                    valores = [np.nan] + valores
                
                return valores

            # Aplicar la función a cada año de siembra en este departamento
            df_semillas.loc[mask, variable] = df_semillas.loc[mask, 'anio'].apply(obtener_clima_previo)

    return df_semillas


# -----------------------------
# PROGRAMA PRINCIPAL
# -----------------------------
años_atras_ejemplo = 44

# Recupero todos los datos de las semillas desde min_year en adelante
df_semillas = pd.read_csv('Recuperacion_de_datos/Semillas/Archivos generados/semillas_todas_concatenadas.csv')

#print("DataFrame semillas original:")
#print(f"Forma: {df_semillas.shape}")
#print(f"AAAAAAAAAAAAAAAAAAAAAAA --- Tipo de anio: {df_semillas['anio'].dtype}")
#print(f"Primeros valores de anio: {df_semillas['anio'].head()}")

# Asegurar que las semillas tengan al menos un año de datos climáticos previos
df_semillas = df_semillas[df_semillas['anio'] >= 1983] # Aseguramos que las semillas tengan al menos un año de datos climáticos previos.

print(f"Semillas después del filtro de años: {df_semillas.shape}")
df_semillas.to_csv('df_semillas_filtradas.csv', index=False)
# ALGO ESTA MAL ACA

df_suelo = pd.read_csv('Recuperacion_de_datos/Suelos/suelo_unido.csv')

# Debo limpiar las columnas con muchos datos faltantes porque interfieren mucho en el entreno de la RN.
# Elimino las columnas que tienen 85% o más datos faltantes
df_suelo_filtrado = df_suelo.dropna(thresh=len(df_suelo) * 0.15, axis=1).copy()
df_semillas_filtrado = df_semillas.dropna(thresh=len(df_semillas) * 0.15, axis=1).copy()

print(f"Semillas después del filtro de columnas: {df_semillas_filtrado.shape}")

path_archivo_suelo = Path("Recuperacion_de_datos/Suelos/suelo_promedio.csv")

if path_archivo_suelo.exists():
    df_suelo_promedio = pd.read_csv('Recuperacion_de_datos/Suelos/suelo_promedio.csv')
else:
    # Crear nueva columna 'departamento' indicando a que departamento corresponde cada dato de suelo
    df_suelo_filtrado.loc[:, 'departamento_nombre'] = df_suelo_filtrado.apply(
        lambda row: solo_departamento([(row['longitude'], row['latitude'])]),
        axis=1
    )
    # Reordenar para que 'departamento' quede como primera columna
    cols = ['departamento_nombre'] + [col for col in df_suelo_filtrado.columns if col != 'departamento_nombre']
    df_suelo_filtrado = df_suelo_filtrado[cols]

    df_suelo_filtrado['coords'] = list(zip(df_suelo_filtrado['longitude'], df_suelo_filtrado['latitude']))

    df_suelo_filtrado.to_csv('Recuperacion_de_datos/Suelos/suelo_filtrado.csv', index=False)

    # Calcular el promedio de las columnas numéricas para cada departamento y centroide para las coordenadas
    df_suelo_promedio = df_suelo_filtrado.groupby('departamento_nombre').agg({
        'coords': lambda x: calcular_centroide(list(x)),   # centroide de todas las coords
        **{col: 'mean' for col in df_suelo_filtrado.columns if col not in ['departamento_nombre', 'coords', 'latitude', 'longitude']}
    }).reset_index()

    df_suelo_promedio.to_csv('Recuperacion_de_datos/Suelos/suelo_promedio.csv', index=False)

print(f"Departamentos en suelo_promedio: {len(df_suelo_promedio)}")
print(f"Departamentos únicos en semillas: {df_semillas_filtrado['departamento_nombre'].nunique()}")

# Recupero clima para cada centroide de suelo_promedio
n_meses = 18
df_semillas_con_clima = agregar_clima_por_departamento(
    df_semillas_filtrado, df_suelo_promedio, años_atras_ejemplo, n_meses
)

print(f"Semillas con clima: {df_semillas_con_clima.shape}")

df_final = pd.merge(
    df_semillas_con_clima,
    df_suelo_promedio,
    on='departamento_nombre',
    how='inner'
)

print(f"DataFrame final: {df_final.shape}")

df_final.to_csv('df_semillas_suelo_clima.csv', index=False)
print("Archivo guardado exitosamente!")