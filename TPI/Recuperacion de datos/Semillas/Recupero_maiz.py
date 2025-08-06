import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
maiz_data = pd.read_csv('Bases de datos/Semillas/maiz-serie-1923-2023.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
mani_data = maiz_data.applymap(limpiar_texto)

maiz_collumns = maiz_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(maiz_collumns.head())

# Guardar
maiz_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/maiz_recuperado.csv', index=False)