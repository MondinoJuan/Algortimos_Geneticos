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
import joblib

def limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    # Definimos los valores a considerar como "inválidos"
    invalid_values = ["SD", "sin datos", "", " ", "null", "NULL", "NaN", "nan", None]

    # Reemplazamos esos valores por NaN
    df_limpio = df.replace(invalid_values, pd.NA)

    # Eliminamos filas con al menos un NaN
    df_limpio = df_limpio.dropna(how="any")

    return df_limpio

def guardar_modelo(modelo, ruta="Modelos_GB/modelo_gbm_viento.pkl"):
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en {ruta}")

def separar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza la columna 'fecha' (formato YYYY-MM-DD)
    por dos columnas: 'anio' y 'mes'.
    """
    # Convertir a datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

    # Crear nuevas columnas
    df['anio'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month

    # Eliminar la columna original
    df = df.drop(columns=['fecha'])

    return df


def entrenar_y_devolver_modelo(head_cant = None):
    # DATA PREPARATION
    df = pd.read_csv("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv")

    if head_cant is not None:
        df = df.head(head_cant)  # Usar solo las primeras head_cant filas para pruebas rápidas

    df_convertido = limpiar_df(df)

    df_convertido = separar_fecha(df_convertido)

    df_train, df_test = np.split(df_convertido, [int(0.8*len(df_convertido))])

    #X_train = df_train[['anio', 'mes', 'velocidad_viento_m_s']]
    X_train = df_train[['anio', 'mes']]

    #X_test = df_test[['anio', 'mes', 'velocidad_viento_m_s']]
    X_test = df_test[['anio', 'mes']]

    y_train = df_train['velocidad_viento_m_s']
    y_test = df_test['velocidad_viento_m_s']

    # BASELINE MODEL
    #crossvalidation = KFold(n_splits = 10, shuffle = True, random_state = 1)
    crossvalidation = KFold(n_splits = 5, shuffle = True, random_state = 1)

    for depth in range(1, 10):
        tree_regressor = tree.DecisionTreeRegressor(max_depth = depth, random_state = 1)
        if tree_regressor.fit(X_train, y_train).tree_.max_depth < depth:
            break
        score = np.mean(cross_val_score(tree_regressor, X_train, y_train,
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

        search.fit(X_train, y_train)

        print("Best params:", search.best_params_)
        print("Best score:", search.best_score_)

        # Paso los mejores parámetros
        GBR2 = GradientBoostingRegressor(**search.best_params_)
    else:
        # GRADIENT BOOSTING MODEL DEVELOPMENT
        # Cambiar luego del hyperparameter tuning
        GBR2 = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.05,
                                        max_depth = 3, subsample = 1, random_state = 1)
        score = np.mean(cross_val_score(GBR2, X_train, y_train,
                                        scoring = 'neg_mean_squared_error',
                                        cv = crossvalidation, n_jobs = 1))
        print("Final model score:", score)

    # PREDECIR
    GBR2.fit(X_train, y_train)
    predicciones = GBR2.predict(X_test)
    #predicciones = GBR2.predict(X_train)
    #guardar_modelo(GBR2)


    # GRÁFICAS
    plt.figure(figsize=(10, 6))

    # Línea azul: todos los valores reales (train + test)
    plt.plot(df_convertido['velocidad_viento_m_s'].values, label='Valores Reales', color='blue', alpha=0.6)

    # Línea roja: predicciones, alineadas con el 20% final
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), predicciones, label='Predicciones', color='red', alpha=0.6)
    #plt.plot(predicciones, label='Predicciones', color='red', alpha=0.6)

    plt.title('Producción Real vs Predicha')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Producción (tn)')
    plt.legend()
    plt.show()

    return GBR2


# -------------------------------
# MAIN
# -------------------------------
def main(entradas_para_predecir = None):
    if entradas_para_predecir is None:
        head_cant = int(input("¿Cuántas filas usar para entrenar? (0 para todas): "))
        if head_cant == 0:
            head_cant = None
        modelo = entrenar_y_devolver_modelo(head_cant)
    else:
        # Cargar modelo guardado
        #modelo = joblib.load("Modelos_GB/modelo_gbm_viento.pkl")
        predicciones = modelo.predict(entradas_para_predecir)

        return predicciones