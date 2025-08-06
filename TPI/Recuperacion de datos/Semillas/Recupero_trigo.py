import pandas as pd
from limpio_caracteres import limpiar_texto

# Leer el archivo en latin1
trigo_data = pd.read_csv('Bases de datos/Semillas/trigo-serie-1927-2024.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
trigo_data = trigo_data.applymap(limpiar_texto)

# Seleccionar columnas
trigo_columnas = trigo_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha',
                             'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]

# Mostrar resultado
print(trigo_columnas.head())

# Guardar archivo limpio
trigo_columnas.to_csv('Recuperacion de datos/Semillas/Archivos generados/trigo_recuperado.csv', index=False, encoding='utf-8')
