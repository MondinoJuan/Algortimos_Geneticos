from Pred_Clima.pred_viento_m_s import main as train_LSTM_viento_ms
from Pred_Clima.pred_viento_km_h import train_LSTM_viento_kmh
from Pred_Clima.pred_temp import train_LSTM_temp
from Pred_Clima.pred_humedad import train_LSTM_humedad
from Pred_Clima.pred_precip import train_LSTM_precip
from Pred_Clima.pred_viento_GBR import entrenar_y_devolver_modelo, main as prediccion_GBR
from Pred_Clima.pred_viento_GBR1 import entrenar_y_devolver_modelo as entrenar_y_devolver_modelo1, main as prediccion_GBR1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv"

df_clima = pd.read_csv(path)
#train_LSTM_temp(df_clima)
#train_LSTM_viento_kmh(df_clima)
#train_LSTM_viento_ms(df_clima)
#train_LSTM_viento_ms(path)
#train_LSTM_humedad(df_clima)
#train_LSTM_precip(df_clima)
#modelo = entrenar_y_devolver_modelo()
#prediccion_GBR(df_clima['velocidad_viento_m_s'])
#prediccion_GBR(df_clima['velocidad_viento_m_s'].values.reshape(-1,1))







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
    df = df.drop(columns=['fecha'])

    return df


# DATA PREPARATION
df_convertido = limpiar_df(df_clima)
df_convertido = separar_fecha(df_convertido)

# Crear columna lag: viento del mes anterior
df_convertido['viento_lag1'] = df_convertido['velocidad_viento_m_s'].shift(1)

# Eliminar la primera fila (porque lag1 queda NaN en el inicio)
df_convertido = df_convertido.dropna()

modelo = entrenar_y_devolver_modelo1()

#predicciones = []
predicciones = prediccion_GBR1(df_convertido[['anio', 'mes', 'viento_lag1']], modelo)

'''for i in range(14):
    pred = prediccion_GBR1(df_convertido[['anio', 'mes', 'viento_lag1']], modelo)
    #prediccion_GBR1(df_convertido[['anio', 'mes', 'viento_lag1']].values.reshape(-1,1), modelo)
    predicciones.append(pred)'''

#predicciones_flat = np.array(predicciones).flatten()

plt.figure(figsize=(10, 6))
plt.plot(df_convertido['velocidad_viento_m_s'].values, label='Valores Reales', color='blue', alpha=0.6)

#plt.plot(range(len(y_train), len(y_train)+len(y_test)), predicciones, label='Predicciones', color='red', alpha=0.6)
plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)
#plt.plot(predicciones_flat, label='Predicciones', color='red', alpha=0.6)

plt.title('Producción Real vs Predicha')
plt.xlabel('Índice de Muestra')
plt.ylabel('Velocidad Viento (m/s)')
plt.legend()
plt.show()
