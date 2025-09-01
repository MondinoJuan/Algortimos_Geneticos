from Pred_Clima.pred_temp import train_LSTM_temp
from Pred_Clima.pred_humedad import train_LSTM_humedad
from Pred_Clima.pred_precip import train_LSTM_precip
from Pred_Clima.pred_viento_GBR import entrenar_y_devolver_modelo as entrenar_y_devolver_modelo1, main as prediccion_GBR1
import pandas as pd
import matplotlib.pyplot as plt

path = "Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv"

df_clima = pd.read_csv(path)
#train_LSTM_temp(df_clima)
#train_LSTM_humedad(df_clima)
#train_LSTM_precip(df_clima)


# DATA PREPARATION
modelo = entrenar_y_devolver_modelo1()

predicciones = prediccion_GBR1(df_clima, modelo)

plt.figure(figsize=(10, 6))
plt.plot(df_clima['velocidad_viento_m_s'].values, label='Valores Reales', color='blue', alpha=0.6)

#plt.plot(range(len(y_train), len(y_train)+len(y_test)), predicciones, label='Predicciones', color='red', alpha=0.6)
plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)

plt.title('Producción Real vs Predicha')
plt.xlabel('Índice de Muestra')
plt.ylabel('Velocidad Viento (m/s)')
plt.legend()
plt.show()
