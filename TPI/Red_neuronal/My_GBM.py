import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np



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


def limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    # Definimos los valores a considerar como "inválidos"
    invalid_values = ["SD", "sin datos", "", " ", "null", "NULL", "NaN", "nan", None]

    # Reemplazamos esos valores por NaN
    df_limpio = df.replace(invalid_values, pd.NA)

    # Eliminamos filas con al menos un NaN
    df_limpio = df_limpio.dropna(how="any")

    return df_limpio





# -------------------------------
# MAIN
# -------------------------------

# DATA PREPARATION
df = pd.read_csv("df_prod_expandido.csv")

df = df.head(10000)  # Usar solo las primeras 100 filas para pruebas rápidas

df_convertido, cultivo_a_entero = conversor_cultivo_a_entero(df)
#df.to_csv("df_semillas_suelo_clima_convertido.csv", index=False)



df_convertido = limpiar_df(df_convertido)

X = df_convertido[['cultivo_nombre', 'anio', 'organic_carbon', 'ph', 'clay', 'silt', 'sand', 
                    'temperatura_media_C_1', 'temperatura_media_C_2', 'temperatura_media_C_3', 'temperatura_media_C_4', 
                    'temperatura_media_C_5', 'temperatura_media_C_6', 'temperatura_media_C_7', 'temperatura_media_C_8', 
                    'temperatura_media_C_9', 'temperatura_media_C_10', 'temperatura_media_C_11', 'temperatura_media_C_12', 
                    'temperatura_media_C_13', 'temperatura_media_C_14', 'humedad_relativa_%_1', 'humedad_relativa_%_2', 
                    'humedad_relativa_%_3', 'humedad_relativa_%_4', 'humedad_relativa_%_5', 'humedad_relativa_%_6', 
                    'humedad_relativa_%_7', 'humedad_relativa_%_8', 'humedad_relativa_%_9', 'humedad_relativa_%_10', 
                    'humedad_relativa_%_11', 'humedad_relativa_%_12', 'humedad_relativa_%_13', 'humedad_relativa_%_14', 
                    'velocidad_viento_m_s_1', 'velocidad_viento_m_s_2', 'velocidad_viento_m_s_3', 'velocidad_viento_m_s_4', 
                    'velocidad_viento_m_s_5', 'velocidad_viento_m_s_6', 'velocidad_viento_m_s_7', 'velocidad_viento_m_s_8', 
                    'velocidad_viento_m_s_9', 'velocidad_viento_m_s_10', 'velocidad_viento_m_s_11', 'velocidad_viento_m_s_12', 
                    'velocidad_viento_m_s_13', 'velocidad_viento_m_s_14', 'velocidad_viento_km_h_1', 'velocidad_viento_km_h_2', 
                    'velocidad_viento_km_h_3', 'velocidad_viento_km_h_4', 'velocidad_viento_km_h_5', 'velocidad_viento_km_h_6', 
                    'velocidad_viento_km_h_7', 'velocidad_viento_km_h_8', 'velocidad_viento_km_h_9', 'velocidad_viento_km_h_10', 
                    'velocidad_viento_km_h_11', 'velocidad_viento_km_h_12', 'velocidad_viento_km_h_13', 'velocidad_viento_km_h_14', 
                    'precipitacion_mm_mes_1', 'precipitacion_mm_mes_2', 'precipitacion_mm_mes_3', 'precipitacion_mm_mes_4', 
                    'precipitacion_mm_mes_5', 'precipitacion_mm_mes_6', 'precipitacion_mm_mes_7', 'precipitacion_mm_mes_8', 
                    'precipitacion_mm_mes_9', 'precipitacion_mm_mes_10', 'precipitacion_mm_mes_11', 'precipitacion_mm_mes_12', 
                    'precipitacion_mm_mes_13', 'precipitacion_mm_mes_14', 'superficie_sembrada_ha']]

y = df_convertido['produccion_tn']

# BASELINE MODEL
#crossvalidation = KFold(n_splits = 10, shuffle = True, random_state = 1)
crossvalidation = KFold(n_splits = 5, shuffle = True, random_state = 1)

for depth in range(1, 10):
    tree_regressor = tree.DecisionTreeRegressor(max_depth = depth, random_state = 1)
    if tree_regressor.fit(X, y).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(tree_regressor, X, y,
                                    scoring='neg_mean_squared_error',
                                    cv=crossvalidation,n_jobs=1))

    print("Depth:", depth, "   -  Score:", score)


# HYPERPARAMETER TUNING
# Solo ejecutar si se cambian los parámetros
decition = input("¿Hacer hyperparameter tuning? (s/n): ")
if decition.lower() == 's':
    print("Iniciando hyperparameter tuning...") 
    GBR = GradientBoostingRegressor()
    #search_grid = {'n_estimators':[500, 1000, 2000], 'learning_rate':[.001, 0.01, .1], 'max_depth':[1, 2, 4], 'subsample':[.5, .75, 1], 'random_state':[1]}
    search_grid = {'n_estimators':[500, 1000], 'learning_rate':[.001, 0.05], 'max_depth':[1, 3], 'subsample':[.5, 1], 'random_state':[1]}
    search = GridSearchCV(estimator = GBR, param_grid = search_grid,
                        scoring = 'neg_mean_squared_error', n_jobs = 1, cv = crossvalidation)

    #search.fit(X, y)

    # tqdm para mostrar progreso
    with tqdm(total=np.prod([len(v) for v in search_grid.values()]), desc="GridSearch") as pbar:
        search.fit(X, y)
        pbar.update(np.prod([len(v) for v in search_grid.values()]))

    print("Best params:", search.best_params_)
    print("Best score:", search.best_score_)

# GRADIENT BOOSTING MODEL DEVELOPMENT
# Cambiar luego de la primer corrida
GBR2 = GradientBoostingRegressor(n_estimators = 2000, learning_rate = 0.01,
                                 max_depth = 2, subsample = 0.5, random_state = 1)
score = np.mean(cross_val_score(GBR2, X, y,
                                scoring = 'neg_mean_squared_error',
                                cv = crossvalidation, n_jobs = 1))
print("Final model score:", score)

# PREDECIR
GBR2.fit(X, y)
predicciones = GBR2.predict(X)


# GRÁFICAS
plt.figure(figsize=(10, 6))
plt.plot(y.values, label='Valores Reales', color='blue', alpha=0.6)
plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)
plt.title('Producción Real vs Predicha')
plt.xlabel('Índice de Muestra')
plt.ylabel('Producción (tn)')
plt.legend()
plt.show()



#df_convertido = pd.read_csv("df_semillas_suelo_clima_convertido.csv")
df_reconvertido = conversor_entero_a_cultivo(df_convertido, cultivo_a_entero)
#df_reconvertido.to_csv("df_semillas_suelo_clima_reconvertido.csv", index=False)