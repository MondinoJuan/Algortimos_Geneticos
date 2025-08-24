# ------------------------------------------------------------------------------------------------------------------------------------------------------
# En este trabajo se realizará una estimación de qué combinación de semillas es la más adecuada para un campo determinado, teniendo en cuenta 
# las características del suelo y el clima. Para ello, se utilizará una red neuronal GBM que tomará como entrada las características climáticas 
# de los últimos 18 meses previos a la siembra, las características del suelo y una mezcla de semillas (que no se normaliza). La salida será 
# la producción estimada en kg.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

from Mapa.GIS.gis_optimizado import MapApp
from creacion_de_archivo_de_entreno_completo import calcular_centroide
from Red_neuronal.My_GBM import main as entrenar_GBM



def __init__():
    # INVOC0 A LA SELECCION DE CAMPO
    '''Devuelve las coordenadas del punto central del campo seleccionado, el nombre del departamento al que pertenece y 
    los metros cuadrados que tiene'''
    app = MapApp()
    app.run()

    longitud, latitud = calcular_centroide(app.coordenadas)
    departamento = app.departamento
    metros_cuadrados = app.area_m2

    # ENTRENO GBM
    entrenar_GBM()

    # EJECUTO EL AG CON EL GBM ENTRENADO COMO FUNCIÓN OBJETIVO