# ------------------------------------------------------------------------------------------------------------------------------------------------------
# En este trabajo se realizará una estimación de qué combinación de semillas es la más adecuada para un campo determinado, teniendo en cuenta 
# las características del suelo y el clima. Para ello, se utilizará una red neuronal GBM que tomará como entrada las características climáticas 
# de los últimos 18 meses previos a la siembra, las características del suelo y una mezcla de semillas (que no se normaliza). La salida será 
# la producción estimada en kg.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import os

from Mapa.GIS.gis_optimizado import MapApp, correr_app
from creacion_de_archivo_de_entreno_completo import calcular_centroide
from Red_neuronal.My_GBM import main as entrenar_GBM
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import main as creo_clima_nasa
from Algoritmo_Genetico.ag import main as ejecuto_ag
from Pred_Clima.pred_temp import train_LSTM_temp
from Pred_Clima.pred_humedad import train_LSTM_humedad
from Pred_Clima.pred_precip_GBR import train_GBR_precip
from Pred_Clima.pred_viento_GBR import main as train_GBR_viento_ms
from funciones_auxiliares_csv import create_df_with_differents_outputs, expand_array_columns
from creacion_de_archivo_de_entreno_completo import main as creacion_archivo_entreno



# INVOC0 A LA SELECCION DE CAMPO
'''Devuelve las coordenadas del punto central del campo seleccionado, el nombre del departamento al que pertenece y 
los metros cuadrados que tiene'''
#app = MapApp()
#app.run()
app = correr_app()

input("Despues de correr app")

longitud, latitud = calcular_centroide(app.coordenadas)
departamento = app.departamento
metros_cuadrados = app.area_m2

# Entreno LSTM
if not os.path.exists("model/model.keras"):
    if not os.path.exists("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv"):
        from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import main as creo_archivo_clima
        creo_archivo_clima(latitud, longitud)

    df_clima = pd.read_csv("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv")
    train_LSTM_temp(df_clima)
    train_GBR_viento_ms()
    train_LSTM_humedad(df_clima)
    train_GBR_precip()

input("Antes de crear archivos")
# Crear un nuevo df_semillas_suelo_clima
creacion_archivo_entreno()

# Crear nuevo df_prod
create_df_with_differents_outputs()

# Crear df_prod_expandido.csv
expand_array_columns()
input("Despues de crear archivos")


# ENTRENO GBM
entrenar_GBM()

# EJECUTO EL AG CON EL GBM ENTRENADO COMO FUNCIÓN OBJETIVO
ejecuto_ag(departamento, longitud, latitud, metros_cuadrados)


print("FIN DEL PROGRAMA")