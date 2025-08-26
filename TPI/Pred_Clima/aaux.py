from Pred_Clima.pred_viento_m_s import main as train_LSTM_viento_ms
from Pred_Clima.pred_viento_km_h import train_LSTM_viento_kmh
import pandas as pd

path = "Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv"

df_clima = pd.read_csv(path)
#train_LSTM_temp(df_clima)
#train_LSTM_viento_kmh(df_clima)
#train_LSTM_viento_ms(df_clima)
train_LSTM_viento_ms(path)
#train_LSTM_humedad(df_clima)
#train_LSTM_precip(df_clima)