import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
mijo_data = pd.read_csv('Bases de datos/Semillas/mijo-serie-1935-2023.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
mijo_data = mijo_data.applymap(limpiar_texto)

mijo_collumns = mijo_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tn', 'rendimiento_kgxha']]

# Mostrar resultado
print(mijo_collumns.head())

# Guardar
mijo_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/mijo_recuperado.csv', index=False)