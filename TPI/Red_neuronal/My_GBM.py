import pandas as pd







# Convertir la columna cultivo_nombre en un entero que represente cada cultivo
def conversor_cultivo_a_entero(df):
    cultivos = df['cultivo_nombre'].unique()
    cultivo_a_entero = {cultivo: i for i, cultivo in enumerate(cultivos)}
    df['cultivo_nombre'] = df['cultivo_nombre'].map(cultivo_a_entero)
    return df, cultivo_a_entero

def conversor_entero_a_cultivo(df, cultivo_a_entero):
    entero_a_cultivo = {i: cultivo for cultivo, i in cultivo_a_entero.items()}
    df['cultivo_nombre'] = df['cultivo_nombre'].map(entero_a_cultivo)
    return df








# -------------------------------
# MAIN
# -------------------------------

df = pd.read_csv("df_semillas_suelo_clima.csv")
df_convertido, cultivo_a_entero = conversor_cultivo_a_entero(df)
#df.to_csv("df_semillas_suelo_clima_convertido.csv", index=False)

#df_convertido = pd.read_csv("df_semillas_suelo_clima_convertido.csv")
df_reconvertido = conversor_entero_a_cultivo(df_convertido, cultivo_a_entero)
#df_reconvertido.to_csv("df_semillas_suelo_clima_reconvertido.csv", index=False)