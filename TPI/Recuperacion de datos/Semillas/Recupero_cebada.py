import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
cebada_data = pd.read_csv('Bases de datos/Semillas/cebada-total-serie-1938-2024.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
cebada_data = cebada_data.applymap(limpiar_texto)

cebada_collumns = cebada_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(cebada_collumns.head())

# Guardar
cebada_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/cebada_recuperado.csv', index=False)