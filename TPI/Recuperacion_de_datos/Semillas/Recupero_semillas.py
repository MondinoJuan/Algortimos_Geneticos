import pandas as pd
from limpio_caracteres import limpiar_texto

# ------------------------------------ AVENA ------------------------------------
def recupero_datos_avena():
    # Load the dataset
    avena_data = pd.read_csv('Bases_de_datos/Semillas/avena-serie-1923-2024.csv', encoding='latin1')
    avena_data = avena_data.applymap(limpiar_texto)
    avena_collumns = avena_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                                 'produccion_tm', 'rendimiento_kgxha']]
    print(avena_collumns.head())
    avena_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/avena_recuperado.csv', index=False)


# ------------------------------------ CEBADA -----------------------------------
def recupero_datos_cebada():
    # Load the dataset
    cebada_data = pd.read_csv('Bases_de_datos/Semillas/cebada-total-serie-1938-2024.csv', encoding='latin1')
    cebada_data = cebada_data.applymap(limpiar_texto)
    cebada_collumns = cebada_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                                   'produccion_tm', 'rendimiento_kgxha']]
    print(cebada_collumns.head())
    cebada_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/cebada_recuperado.csv', index=False)

# ------------------------------------ CENTENO -----------------------------------
def recupero_datos_centeno():
    # Load the dataset
    centeno_data = pd.read_csv('Bases_de_datos/Semillas/centeno-serie-1923-2024.csv', encoding='latin1')
    centeno_data = centeno_data.applymap(limpiar_texto)
    centeno_collumns = centeno_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                                     'produccion_tn', 'rendimiento_kgxha']]
    print(centeno_collumns.head())
    centeno_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/centeno_recuperado.csv', index=False)

# ------------------------------------ GIRASOL -----------------------------------
def recupero_datos_girasol():
    # Load the dataset
    girasol_data = pd.read_csv('Bases_de_datos/Semillas/girasol-serie-1969-2023.csv', encoding='latin1')
    girasol_data = girasol_data.applymap(limpiar_texto)
    girasol_collumns = girasol_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                                     'produccion_tm', 'rendimiento_kgxha']]
    print(girasol_collumns.head())
    girasol_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/girasol_recuperado.csv', index=False)

# ------------------------------------ MAIZ -----------------------------------
def recupero_datos_maiz():
    # Load the dataset
    maiz_data = pd.read_csv('Bases_de_datos/Semillas/maiz-serie-1923-2023.csv', encoding='latin1')
    maiz_data = maiz_data.applymap(limpiar_texto)
    maiz_collumns = maiz_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                               'produccion_tm', 'rendimiento_kgxha']]
    print(maiz_collumns.head())
    maiz_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/maiz_recuperado.csv', index=False)

# ------------------------------------ MANI -----------------------------------
def recupero_datos_mani():
    # Load the dataset
    mani_data = pd.read_csv('Bases_de_datos/Semillas/mani-serie-1927-2023.csv', encoding='latin1')
    mani_data = mani_data.applymap(limpiar_texto)
    mani_collumns = mani_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                               'produccion_tm', 'rendimiento_kgxha']]
    print(mani_collumns.head())
    mani_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/mani_recuperado.csv', index=False)

# ------------------------------------ MIJO -----------------------------------
def recupero_datos_mijo():
    # Load the dataset
    mijo_data = pd.read_csv('Bases_de_datos/Semillas/mijo-serie-1935-2023.csv', encoding='latin1')
    mijo_data = mijo_data.applymap(limpiar_texto)
    mijo_collumns = mijo_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                               'produccion_tn', 'rendimiento_kgxha']]
    print(mijo_collumns.head())
    mijo_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/mijo_recuperado.csv', index=False)

# ------------------------------------ SOJA -----------------------------------
def recupero_datos_soja():
    # Load the dataset
    soja_data = pd.read_csv('Bases_de_datos/Semillas/soja-serie-1941-2023.csv', encoding='latin1')
    soja_data = soja_data.applymap(limpiar_texto)
    soja_collumns = soja_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha', 'superficie_cosechada_ha', 
                               'produccion_tm', 'rendimiento_kgxha']]
    print(soja_collumns.head())
    soja_collumns.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/soja_recuperado.csv', index=False)

# ------------------------------------ TRIGO -----------------------------------
def recupero_datos_trigo():
   # Leer el archivo en latin1
    trigo_data = pd.read_csv('Bases_de_datos/Semillas/trigo-serie-1927-2024.csv', encoding='latin1')
    trigo_data = trigo_data.applymap(limpiar_texto)
    trigo_columnas = trigo_data[['anio', 'departamento_nombre', 'superficie_sembrada_ha',
                                'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']]
    print(trigo_columnas.head())
    trigo_columnas.to_csv('Recuperacion_de_datos/Semillas/Archivos generados/trigo_recuperado.csv', 
                          index=False, encoding='utf-8')