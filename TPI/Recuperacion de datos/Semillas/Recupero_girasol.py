import pandas as pd
from limpio_caracteres import limpiar_texto

# Load the dataset
girasol_data = pd.read_csv('Bases de datos/Semillas/girasol-serie-1969-2023.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
girasol_data = girasol_data.applymap(limpiar_texto)

girasol_collumns = girasol_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(girasol_collumns.head())

# Guardar
girasol_collumns.to_csv('Recuperacion de datos/Semillas/Archivos generados/girasol_recuperado.csv', index=False)