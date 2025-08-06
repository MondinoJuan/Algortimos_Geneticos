import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
soja_data = pd.read_csv('Bases de datos/Semillas/soja-serie-1941-2023.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
soja_data = soja_data.applymap(limpiar_texto)

soja_collumns = soja_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(soja_collumns.head())

# Guardar
soja_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/soja_recuperado.csv', index=False)