import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
avena_data = pd.read_csv('Bases de datos/Semillas/avena-serie-1923-2024.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
avena_data = avena_data.applymap(limpiar_texto)

avena_collumns = avena_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(avena_collumns.head())

# Guardar
avena_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/avena_recuperado.csv', index=False)