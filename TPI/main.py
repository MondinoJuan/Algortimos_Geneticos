# ------------------------------------------------------------------------------------------------------------------------------------------------------
# En este trabajo se realizará una estimación de qué combinación de semillas es la más adecuada para un campo determinado, teniendo en cuenta 
# las características del suelo y el clima. Para ello, se utilizará una red neuronal LSTM que tomará como entrada las características climáticas 
# de los últimos 18 meses previos a la siembra, las características del suelo y una mezcla de semillas (que no se normaliza). La salida será 
# la producción estimada en kg.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd



def __init__():
    # INVOC0 A LA SELECCION DE CAMPO
    '''Devuelve las coordenadas del punto central del campo seleccionado, el nombre del departamento al que pertenece y los metros cuadrados que tiene'''
    latitud, longitud, departamento, metros_cuadrados = 0, 0, '', 0

    # Leo archivo .csv con todos los datos de clima, suelo y semillas
    


    # ENTRENO LSTM
    '''Entreno la red neuronal con los datos obtenidos y la guardo para su uso posterior'''

    # EJECUTO EL AG CON EL LSTM ENTRENADO COMO FUNCIÓN OBJETIVO