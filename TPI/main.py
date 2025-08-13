# ------------------------------------------------------------------------------------------------------------------------------------------------------
# En este trabajo se realizará una estimación de qué combinación de semillas es la más adecuada para un campo determinado, teniendo en cuenta 
# las características del suelo y el clima. Para ello, se utilizará una red neuronal LSTM que tomará como entrada las características climáticas 
# de los últimos 18 meses previos a la siembra, las características del suelo y una mezcla de semillas (que no se normaliza). La salida será 
# la producción estimada en kg.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

from Recuperacion_de_datos.Suelos.Recupero_suelos_de_tablas import recupero_datos_suelo
from Recuperacion_de_datos.Semillas.Recupero_semillas import recupero_datos_avena, recupero_datos_cebada, recupero_datos_centeno, \
    recupero_datos_girasol, recupero_datos_maiz, recupero_datos_mani, recupero_datos_mijo, recupero_datos_soja, recupero_datos_trigo
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import obtener_datos_nasa_power, procesar_datos_mensuales
from Red_neuronal.utilities import climate_for_coordinates, suelo_for_coordinates, seeds_for_department
import pandas as pd


semillas = ['avena', 'cebada', 'centeno', 'girasol', 'maiz', 'mani', 'mijo', 'soja', 'trigo']   # Podriamos implementar que el usuario pueda decidir 
                                                                                                # las semillas a utilizar


def __init__():
    # INVOC0 A LA SELECCION DE CAMPO
    '''Devuelve las coordenadas del punto central del campo seleccionado, el nombre del departamento al que pertenece y los metros cuadrados que tiene'''
    latitud, longitud, departamento, metros_cuadrados = 0, 0, '', 0

    # Creo archivo .csv con todos los datos de clima, suelo y semillas
    data_todas_semillas = pd.read_csv('Recuperacion_de_datos/Semillas/Archivos generados/semillas_todas_concatenadas.csv')
    clima_df = obtener_datos_nasa_power(latitud, longitud)
    clima_df = procesar_datos_mensuales(clima_df)
    suelo_df = pd.read_csv('Recuperacion_de_datos/Suelos/suelo_unido.csv')

    '''VER QUE HAGO CON LOS DATOS DE ARRIBA, COMO ENTRENO EL LSTM Y ESO'''

    # ENTRENO LSTM
    '''Entreno la red neuronal con los datos obtenidos y la guardo para su uso posterior'''

    # EJECUTO EL AG CON EL LSTM ENTRENADO COMO FUNCIÓN OBJETIVO