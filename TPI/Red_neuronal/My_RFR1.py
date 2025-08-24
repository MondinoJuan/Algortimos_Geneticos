import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # para guardar/cargar el modelo y encoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ===========================
# 1. Cargar datos
# ===========================
data = pd.read_csv("df_prod_expandido.csv", low_memory=False)

# ===========================
# 2. Features y target
# ===========================
features = [
    'cultivo_nombre','anio','departamento_nombre',
    'organic_carbon','ph','clay','silt','sand',
    'temperatura_media_C_1','temperatura_media_C_2','temperatura_media_C_3','temperatura_media_C_4',
    'temperatura_media_C_5','temperatura_media_C_6','temperatura_media_C_7','temperatura_media_C_8',
    'temperatura_media_C_9','temperatura_media_C_10','temperatura_media_C_11','temperatura_media_C_12',
    'temperatura_media_C_13','temperatura_media_C_14',
    'humedad_relativa_%_1','humedad_relativa_%_2','humedad_relativa_%_3','humedad_relativa_%_4',
    'humedad_relativa_%_5','humedad_relativa_%_6','humedad_relativa_%_7','humedad_relativa_%_8',
    'humedad_relativa_%_9','humedad_relativa_%_10','humedad_relativa_%_11','humedad_relativa_%_12',
    'humedad_relativa_%_13','humedad_relativa_%_14',
    'velocidad_viento_m_s_1','velocidad_viento_m_s_2','velocidad_viento_m_s_3','velocidad_viento_m_s_4',
    'velocidad_viento_m_s_5','velocidad_viento_m_s_6','velocidad_viento_m_s_7','velocidad_viento_m_s_8',
    'velocidad_viento_m_s_9','velocidad_viento_m_s_10','velocidad_viento_m_s_11','velocidad_viento_m_s_12',
    'velocidad_viento_m_s_13','velocidad_viento_m_s_14',
    'velocidad_viento_km_h_1','velocidad_viento_km_h_2','velocidad_viento_km_h_3','velocidad_viento_km_h_4',
    'velocidad_viento_km_h_5','velocidad_viento_km_h_6','velocidad_viento_km_h_7','velocidad_viento_km_h_8',
    'velocidad_viento_km_h_9','velocidad_viento_km_h_10','velocidad_viento_km_h_11','velocidad_viento_km_h_12',
    'velocidad_viento_km_h_13','velocidad_viento_km_h_14',
    'precipitacion_mm_mes_1','precipitacion_mm_mes_2','precipitacion_mm_mes_3','precipitacion_mm_mes_4',
    'precipitacion_mm_mes_5','precipitacion_mm_mes_6','precipitacion_mm_mes_7','precipitacion_mm_mes_8',
    'precipitacion_mm_mes_9','precipitacion_mm_mes_10','precipitacion_mm_mes_11','precipitacion_mm_mes_12',
    'precipitacion_mm_mes_13','precipitacion_mm_mes_14',
    'superficie_sembrada_ha'
]
target = 'produccion_tn'

# ===========================
# 3. Limpiar datos - eliminar filas con NaN, vacíos o 'SD'
# ===========================
# Seleccionar solo las columnas que necesitamos
columns_needed = features + [target]
data_subset = data[columns_needed].copy()

# Reemplazar 'SD' y strings vacíos con NaN
data_subset = data_subset.replace(['SD', '', ' '], np.nan)

# Eliminar filas con cualquier valor NaN
data_clean = data_subset.dropna()

print(f"Filas originales: {len(data)}")
print(f"Filas después de limpiar NaN/SD/vacíos: {len(data_clean)}")

X = data_clean[features]
y = data_clean[target]

# ===========================
# 4. Codificar variables categóricas (One-Hot)
# ===========================
categorical_cols = ['cultivo_nombre','departamento_nombre']
numeric_cols = [c for c in features if c not in categorical_cols]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X[categorical_cols])
X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X_num = X[numeric_cols]
X_encoded = pd.concat([X_num, X_cat_df], axis=1)

# ===========================
# 5. Train / Test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.15, random_state=42
)

# ===========================
# 6. Entrenar modelo Random Forest
# ===========================
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ===========================
# 7. Evaluación
# ===========================
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# ===========================
# 8. Importancia de features
# ===========================
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
print("\nFeature importance:\n", importances.head(15))

# ===========================
# 9. Gráfico y_test vs y_pred
# ===========================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Random Forest - Producción (tn)")
plt.show()

# ===========================
# 10. Guardar modelo y encoder
# ===========================
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\nModelo y encoder guardados en disco.")

# ===========================
# 11. Ejemplo: cargar y usar modelo en datos nuevos
# ===========================
# rf_loaded = joblib.load("random_forest_model.pkl")
# encoder_loaded = joblib.load("encoder.pkl")

# # Preparar un ejemplo nuevo (dict con features)
# nuevo = pd.DataFrame([{
#     "cultivo_nombre": "soja",
#     "anio": 2023,
#     "departamento_nombre": "Rosario",
#     "organic_carbon": 1.2,
#     "ph": 6.5,
#     "clay": 30,
#     "silt": 40,
#     "sand": 30,
#     "temperatura_media_C": 22,
#     "humedad_relativa_%": 65,
#     "velocidad_viento_m_s": 3.5,
#     "velocidad_viento_km_h": 12.6,
#     "precipitacion_mm_mes": 100,
#     "superficie_sembrada_ha": 500
# }])

# # Codificar y predecir
# X_cat_new = encoder_loaded.transform(nuevo[categorical_cols])
# X_cat_df_new = pd.DataFrame(X_cat_new, columns=encoder_loaded.get_feature_names_out(categorical_cols))
# X_num_new = nuevo[numeric_cols].reset_index(drop=True)
# X_new = pd.concat([X_num_new, X_cat_df_new.reset_index(drop=True)], axis=1)

# pred_nuevo = rf_loaded.predict(X_new)
# print("Predicción de producción (tn):", pred_nuevo[0])import pandas as pd