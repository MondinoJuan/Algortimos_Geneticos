import pandas as pd

df_final = pd.read_csv('df_semillas_suelo_clima.csv')
column_order = [
    "cultivo_nombre", "anio", "departamento_nombre", "organic_carbon", "ph", "clay", "silt", "sand", 
    "temperatura_media_C", "humedad_relativa_%", "velocidad_viento_m_s", "velocidad_viento_km_h", 
    "precipitacion_mm_mes", "superficie_sembrada_ha",   # Entradas
    "rendimiento_kgxha"                                 # Salida esperada
]
df_final = df_final[column_order]

df_final.to_csv('df_semillas_suelo_clima.csv', index=False)