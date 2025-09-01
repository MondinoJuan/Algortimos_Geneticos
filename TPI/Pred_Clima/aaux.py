from Pred_Clima.pred_temp import train_LSTM_temp
from Pred_Clima.pred_humedad import train_LSTM_humedad
from Pred_Clima.pred_precip import train_LSTM_precip
from Pred_Clima.pred_precip_GBR import entrenar_y_devolver_modelo as entrenar_y_devolver_modelo1, main as prediccion_GBR1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv"

df_clima = pd.read_csv(path)

def limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    # Definimos los valores a considerar como "inválidos"
    invalid_values = ["SD", "sin datos", "", " ", "null", "NULL", "NaN", "nan", None]

    # Reemplazamos esos valores por NaN
    df_limpio = df.replace(invalid_values, pd.NA)

    # Eliminamos filas con al menos un NaN
    df_limpio = df_limpio.dropna(how="any")

    return df_limpio


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
    #df = df.drop(columns=['fecha'])

    return df


df = limpiar_df(df_clima)
df = separar_fecha(df)
df['viento_lag1'] = df['velocidad_viento_m_s'].shift(1)

#train_LSTM_temp(df_clima)
#train_LSTM_humedad(df_clima)
#train_LSTM_precip(df_clima)


'''
# DATA PREPARATION
modelo = entrenar_y_devolver_modelo1()

#predicciones = prediccion_GBR1(df_clima, modelo)

df_train, df_test = np.split(df_clima, [int(0.8*len(df_clima))])

predicciones = prediccion_GBR1(df_test, modelo)

plt.figure(figsize=(10, 6))
#plt.plot(df_clima['precipitacion_mm_mes'].values, label='Valores Reales', color='blue', alpha=0.6)
plt.plot(df_test['precipitacion_mm_mes'].values, label='Valores Reales', color='blue', alpha=0.6)

#plt.plot(range(len(y_train), len(y_train)+len(y_test)), predicciones, label='Predicciones', color='red', alpha=0.6)
plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)

plt.title('Producción Real vs Predicha')
plt.xlabel('Índice de Muestra')
plt.ylabel('Precipitacion (mm)')
plt.legend()
plt.show()
'''

