import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
mani_data = pd.read_csv('Bases de datos/Semillas/mani-serie-1927-2023.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
mani_data = mani_data.applymap(limpiar_texto)

mani_collumns = mani_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(mani_collumns.head())

# Guardar
mani_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/mani_recuperado.csv', index=False)