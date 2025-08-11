import pandas as pd
from limpio_caracteres import limpiar_texto

# Leer el archivo en latin1
semillas_data = pd.read_csv('Bases_de_datos/Semillas/1_Historico_Semillas.csv', encoding='latin1')

# Aplicar limpieza a todo el DataFrame
semillas_data = semillas_data.applymap(limpiar_texto)

# Seleccionar columnas
semillas_columnas = semillas_data[['departamento_nombre', 'cultivo', 'ciclo', 'sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']]

# Dejar solo el año inicial en la columna 'ciclo'
semillas_columnas['ciclo'] = semillas_columnas['ciclo'].str.split('/').str[0]

# Mostrar resultado
print(semillas_columnas.head())

# Guardar archivo limpio
semillas_columnas.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/semillas_historico_recuperado.csv', index=False, encoding='utf-8')