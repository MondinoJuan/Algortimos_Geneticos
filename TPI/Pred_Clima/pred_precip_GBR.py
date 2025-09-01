import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import joblib

def limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    # Definimos los valores a considerar como "inválidos"
    invalid_values = ["SD", "sin datos", "", " ", "null", "NULL", "NaN", "nan", None]

    # Reemplazamos esos valores por NaN
    df_limpio = df.replace(invalid_values, pd.NA)

    # Eliminamos filas con al menos un NaN
    df_limpio = df_limpio.dropna(how="any")

    return df_limpio

def guardar_modelo(modelo, ruta="model/modelo_gbm_precip.pkl"):
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en {ruta}")

def separar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza la columna 'fecha' (formato YYYY-MM-DD)
    por dos columnas: 'anio' y 'mes'.
    """
    # Convertir a datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

    # Crear nuevas columnas
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month

    # Eliminar la columna original
    df = df.drop(columns=['fecha'])

    return df


def entrenar_y_devolver_modelo(head_cant=None):
    # DATA PREPARATION
    df = pd.read_csv("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv")

    if head_cant is not None:
        df = df.head(head_cant)  # Usar solo las primeras head_cant filas para pruebas rápidas

    df_convertido = limpiar_df(df)
    df_convertido = separar_fecha(df_convertido)

    # Crear columna lag: precipitaciones del mes anterior
    df_convertido['precip_lag1'] = df_convertido['precipitacion_mm_mes'].shift(1)

    # Eliminar la primera fila (porque lag1 queda NaN en el inicio)
    df_convertido = df_convertido.dropna()

    # Dividir train / test
    df_train, df_test = np.split(df_convertido, [int(0.8*len(df_convertido))])

    # Features incluyen año, mes y precipitaciones del mes anterior
    X_train = df_train[['anio', 'mes', 'precip_lag1']]
    X_test = df_test[['anio', 'mes', 'precip_lag1']]

    y_train = df_train['precipitacion_mm_mes']
    y_test = df_test['precipitacion_mm_mes']

    # CROSSVALIDATION
    crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)

    GBR2 = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=3, subsample=1, random_state=1
    )

    score = np.mean(cross_val_score(
        GBR2, X_train, y_train,
        scoring='neg_mean_squared_error',
        cv=crossvalidation, n_jobs=1
    ))
    print("Final model score:", score)

    # Entrenar
    GBR2.fit(X_train, y_train)
    predicciones = GBR2.predict(X_test)
    #predicciones = GBR2.predict(X_train)
    guardar_modelo(GBR2)

    # GRÁFICAS
    plt.figure(figsize=(10, 6))
    plt.plot(df_convertido['precipitacion_mm_mes'].values, label='Valores Reales', color='blue', alpha=0.6)

    #plt.plot(range(len(y_train), len(y_train)+len(y_test)), predicciones, label='Predicciones', color='red', alpha=0.6)
    plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)

    plt.title('Producción Real vs Predicha')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Precipitaciones (mm)')
    plt.legend()
    plt.show()

    return GBR2



# -------------------------------
# MAIN
# -------------------------------
def main(entradas_para_predecir = None, modelo = None):
    if entradas_para_predecir is None:
        head_cant = int(input("¿Cuántas filas usar para entrenar? (0 para todas): "))
        if head_cant == 0:
            head_cant = None
        modelo = entrenar_y_devolver_modelo(head_cant)
    else:
        # Cargar modelo guardado
        modelo = joblib.load("model/modelo_gbm_precip.pkl")

        df = limpiar_df(entradas_para_predecir)
        df = separar_fecha(df)
        df['precip_lag1'] = df['precipitacion_mm_mes'].shift(1)
        df = df.dropna()
        ultima_tupla = df.tail(1)

        predicciones = []

        for i in range(14):
            prediccion = modelo.predict(ultima_tupla[['anio', 'mes', 'precip_lag1']])
            nueva_fila = {
                'anio': ultima_tupla['anio'].values[0] + (ultima_tupla['mes'].values[0] // 12),
                'mes': (ultima_tupla['mes'].values[0] % 12) + 1,
                'precipitacion_mm_mes': prediccion[0],
                'precip_lag1': ultima_tupla['precipitacion_mm_mes'].values[0]
            }
            nueva_fila_df = pd.DataFrame([nueva_fila])
            predicciones.append(prediccion[0])
            ultima_tupla = nueva_fila_df

        return predicciones