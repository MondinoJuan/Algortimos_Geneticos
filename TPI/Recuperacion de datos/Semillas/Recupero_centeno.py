import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
centeno_data = pd.read_csv('Bases de datos/Semillas/centeno-serie-1923-2024.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
centeno_data = centeno_data.applymap(limpiar_texto)

centeno_collumns = centeno_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tn', 'rendimiento_kgxha']]

# Mostrar resultado
print(centeno_collumns.head())

# Guardar
centeno_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/centeno_recuperado.csv', index=False)