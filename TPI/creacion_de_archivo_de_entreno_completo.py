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

#from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import obtener_datos_nasa_power, procesar_datos_mensuales
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import main as recupero_datos_clima_mensual
from Mapa.GIS.departamento import encontrar_departamento as transformo_coord_a_depto


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

# -----------------------------
# Si quiero usar cache

def agregar_clima_por_departamento(df_suelo_semillas, n_meses=14):
    df_suelo_semillas = df_suelo_semillas.copy()

    # Cache de datos climáticos por coordenadas (evita re-llamar a la API)
    clima_cache = {}
    # Cache de "ventanas" ya calculadas: (coords_key, variable, año_siembra) -> lista de valores
    ventana_cache = {}

    total = len(df_suelo_semillas)

    for i, (_, row) in enumerate(df_suelo_semillas.iterrows(), start=1):
        depto = row['departamento_nombre']
        lon_str, lat_str = row['coords'].strip("()").split(",")
        lon, lat = float(lon_str), float(lat_str)

        print(f"[{i}/{total}] Procesando depto={depto} (lon={lon}, lat={lat})")

        # Elegí una clave estable para el cache. Si el 'depto' no cambia la respuesta de la API,
        # podés dejar solo coords; si sí la cambia, incluí depto en la clave.
        coords_key = (round(lon, 5), round(lat, 5))  # o row['coords'] si siempre viene igual

        # Traer/recuperar df_clima_filtrado desde cache o API
        if coords_key in clima_cache:
            df_clima_filtrado = clima_cache[coords_key]
            # print(f"[{i}/{total}] Re-uso clima cacheado para {coords_key} (depto={depto})")
        else:
            # print(f"[{i}/{total}] Consultando clima para {coords_key} (depto={depto})")
            df_clima = recupero_datos_clima_mensual(lon, lat, depto)
            df_clima['fecha'] = pd.to_datetime(df_clima['fecha'], errors='coerce')
            df_clima_filtrado = df_clima.dropna(thresh=len(df_clima) * 0.15, axis=1).copy()
            clima_cache[coords_key] = df_clima_filtrado  # guardo en cache

        # Filtrar semillas de este departamento (si hay varios años para el mismo depto)
        mask = df_suelo_semillas['departamento_nombre'] == depto

        for variable in ['temperatura_media_C', 'humedad_relativa_%', 'velocidad_viento_m_s', 'precipitacion_mm_mes']:

            def obtener_clima_previo(año_siembra):
                # Ver si ya calculamos esta ventana para (coords, variable, año)
                vkey = (coords_key, variable, int(año_siembra))
                if vkey in ventana_cache:
                    return ventana_cache[vkey]

                # Crear fecha límite y seleccionar datos previos
                fecha_limite = pd.Timestamp(year=int(año_siembra) - 1, month=12, day=31)
                clima_previo = df_clima_filtrado[df_clima_filtrado['fecha'] <= fecha_limite]

                if len(clima_previo) == 0:
                    # print(f"  - No hay datos climáticos para {depto} antes de {año_siembra}")
                    ventana_cache[vkey] = []
                    return ventana_cache[vkey]

                valores = clima_previo[variable].tail(n_meses).tolist()
                while len(valores) < n_meses:
                    valores = [np.nan] + valores

                ventana_cache[vkey] = valores  # guardo en cache
                return valores

            # Aplicar por año de siembra para todas las filas de este depto
            df_suelo_semillas.loc[mask, variable] = (
                df_suelo_semillas.loc[mask, 'anio'].apply(obtener_clima_previo)
            )

    return df_suelo_semillas

# -----------------------------

'''def agregar_clima_por_departamento(df_suelo_semillas, n_meses=14):
    df_suelo_semillas = df_suelo_semillas.copy()
    total = len(df_suelo_semillas)

    # Iterar sobre cada departamento en df_suelo_promedio
    for i, (_, row) in enumerate(df_suelo_semillas.iterrows(), start=1):
    #for i, row in enumerate(df_suelo_semillas.itertuples(index=False), start=1):
        depto = row['departamento_nombre']
        #depto = row.departamento_nombre
        lon_str, lat_str = row['coords'].strip("()").split(",")
        #lon_str, lat_str = row.coords.strip("()").split(",")
        lon, lat = float(lon_str), float(lat_str)

        print(f"[{i}/{total}] Procesando depto={depto} (lon={lon}, lat={lat})")
        #print(f"[{i}/{total}] Procesando depto={depto}")

        #df_clima = procesar_datos_mensuales(df_clima_diario)
        df_clima = recupero_datos_clima_mensual(lon, lat, depto)
        df_clima['fecha'] = pd.to_datetime(df_clima['fecha'], errors='coerce')
        df_clima_filtrado = df_clima.dropna(thresh=len(df_clima) * 0.15, axis=1).copy()

        # Filtrar semillas de este departamento
        mask = df_suelo_semillas['departamento_nombre'] == depto

        for variable in ['temperatura_media_C','humedad_relativa_%', 'velocidad_viento_m_s', 'precipitacion_mm_mes']:
            
            def obtener_clima_previo(año_siembra):
                """
                Obtiene los últimos n_meses de datos climáticos anteriores al año de siembra
                """
                # Crear fecha límite: 31 de diciembre del año anterior
                fecha_limite = pd.Timestamp(year=año_siembra-1, month=12, day=31)
                
                # Filtrar datos anteriores a la fecha límite
                clima_previo = df_clima_filtrado[df_clima_filtrado['fecha'] <= fecha_limite]
                
                if len(clima_previo) == 0:
                    print(f"No hay datos climáticos para {depto} antes de {año_siembra}")
                    return []
                
                # Tomar los últimos n_meses y retornar como lista
                valores = clima_previo[variable].tail(n_meses).tolist()
                
                # Si no tenemos suficientes meses, rellenar con NaN
                while len(valores) < n_meses:
                    valores = [np.nan] + valores
                
                return valores

            # Aplicar la función a cada año de siembra en este departamento
            df_suelo_semillas.loc[mask, variable] = df_suelo_semillas.loc[mask, 'anio'].apply(obtener_clima_previo)

    return df_suelo_semillas'''


def crear_df_suelo_semillas(path_suelo_semillas):
    # Recupero todos los datos de las semillas desde min_year en adelante
    df_semillas = pd.read_csv('Recuperacion_de_datos/Semillas/Archivos generados/semillas_todas_concatenadas.csv')

    # Asegurar que las semillas tengan al menos un año de datos climáticos previos
    df_semillas = df_semillas[df_semillas['anio'] >= 1983] # Aseguramos que las semillas tengan al menos un año de datos climáticos previos.

    df_suelo = pd.read_csv('Recuperacion_de_datos/Suelos/suelo_unido.csv')

    # Debo limpiar las columnas con muchos datos faltantes porque interfieren mucho en el entreno de la RN.
    # Elimino las columnas que tienen 85% o más datos faltantes
    df_suelo_filtrado = df_suelo.dropna(thresh=len(df_suelo) * 0.15, axis=1).copy()
    df_semillas_filtrado = df_semillas.dropna(thresh=len(df_semillas) * 0.15, axis=1).copy()

    path_archivo_suelo = Path("Recuperacion_de_datos/Suelos/suelo_promedio.csv")

    if path_archivo_suelo.exists():
        df_suelo_promedio = pd.read_csv(path_archivo_suelo)
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

        df_suelo_promedio.to_csv(path_archivo_suelo, index=False)

    df_suelo_semillas = pd.merge(df_semillas_filtrado, df_suelo_promedio, on = 'departamento_nombre', how = 'inner')
    
    df_suelo_semillas.to_csv(path_suelo_semillas)

    return df_suelo_semillas

# -----------------------------
# PROGRAMA PRINCIPAL
# -----------------------------
def main():
    años_atras_ejemplo = 44

    path_suelo_semillas = 'Archivos/df_suelo_Semillas.csv'

    if not os.path.exists(path_suelo_semillas):
        df_suelo_semillas = crear_df_suelo_semillas(path_suelo_semillas)
    else:
        df_suelo_semillas = pd.read_csv(path_suelo_semillas)

    # Recupero clima para cada centroide de suelo_promedio
    n_meses = 14

    df_final = agregar_clima_por_departamento(df_suelo_semillas, n_meses)

    df_final["cultivo_nombre"] = df_final["cultivo_nombre"].str.strip().str.lower()

    # Como ya no hay merges deberia sacar departamento_nombre y coords
    column_order = [
        "cultivo_nombre", "anio", "departamento_nombre", "coords", "organic_carbon", "ph", "clay", "silt", "sand", 
        "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "precipitacion_mm_mes", "superficie_sembrada_ha",                               # Entradas
        "superficie_cosechada_ha", "produccion_tn", "rendimiento_kgxha"                                                                                      # Salida esperada
    ]
    df_final = df_final[column_order]

    # --- Eliminar filas con datos faltantes o infinitos ---
    # 1) Pasar ±inf a NaN (scikit-learn también los rechaza)
    df_final = df_final.replace([np.inf, -np.inf], np.nan)

    # 2) Marcar filas con al menos un NaN
    mask_nan = df_final.isna().any(axis=1)

    # 3) (Opcional) ver cuántas vas a eliminar
    print("Filas eliminadas por NaN/inf:", int(mask_nan.sum()))

    # 4) Filtrar y resetear índice
    df_final = df_final[~mask_nan].reset_index(drop=True)
    # --- fin limpieza ---

    df_final.to_csv('Archivos/df_semillas_suelo_clima.csv', index=False)
    print("Archivo guardado exitosamente!")

if __name__ == "__main__":
    main()