import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PARÁMETROS IMPORTANTES
# -----------------------------
SEQUENCE_LENGTH = 18  # 18 meses previos
CLIMATE_FEATURES = ['temperatura_media_C', 'humedad_relativa_%',
                    'velocidad_viento_m_s', 'precipitacion_mm_mes']
SOIL_FEATURES = ['bulk_density','ca_co3','coarse_fragments','ecec',
                 'conductivity','organic_carbon','ph','clay','silt','sand','water_retention']
# La combinación de semillas deberías codificarla en forma numérica antes (One-hot o embedding index)

# -----------------------------
# FUNCIÓN PARA PREPARAR DATOS
# -----------------------------
def prepare_dataset(climate_df, soil_df, seed_info_df, target_column='produccion_tm'):
    """
    climate_df: dataframe con columnas de fecha + CLIMATE_FEATURES
    soil_df: dataframe con SOIL_FEATURES (solo una fila por campo/departamento)
    seed_info_df: dataframe con info de semillas y producción histórica
    target_column: 'produccion_tm' (convertir a kg dentro de la función)
    """

    # Normalizar datos climáticos
    scaler_climate = StandardScaler()
    climate_scaled = scaler_climate.fit_transform(climate_df[CLIMATE_FEATURES])

    # Normalizar suelo
    scaler_soil = StandardScaler()
    soil_scaled = scaler_soil.fit_transform(soil_df[SOIL_FEATURES])

    # Convertir producción de toneladas a kg
    y = seed_info_df[target_column].values * 1000  # toneladas → kg

    # Generar secuencias de clima
    X_climate_seq = []
    for i in range(len(climate_scaled) - SEQUENCE_LENGTH):
        seq = climate_scaled[i:i+SEQUENCE_LENGTH]
        X_climate_seq.append(seq)
    X_climate_seq = np.array(X_climate_seq)

    # Expandir suelo para cada secuencia
    soil_repeated = np.repeat(soil_scaled, X_climate_seq.shape[0], axis=0)
    soil_repeated = soil_repeated.reshape(X_climate_seq.shape[0], 1, len(SOIL_FEATURES))

    # Combinar datos de clima + suelo (e.g. concatenar en la dimensión de features)
    X_combined = np.concatenate([X_climate_seq, np.repeat(soil_repeated, SEQUENCE_LENGTH, axis=1)], axis=2)

    # Ajustar tamaño de Y para que coincida
    y = y[SEQUENCE_LENGTH:]

    return X_combined, y, scaler_climate, scaler_soil

# -----------------------------
# CREACIÓN DEL MODELO LSTM
# -----------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Producción en kg
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# -----------------------------
# EJEMPLO DE ENTRENAMIENTO
# -----------------------------
# climate_df = pd.read_csv('clima.csv')  
# soil_df = pd.read_csv('suelo.csv')     
# seed_info_df = pd.read_csv('semillas.csv') 

# X, y, sc_climate, sc_soil = prepare_dataset(climate_df, soil_df, seed_info_df)
# model = build_lstm_model((SEQUENCE_LENGTH, X.shape[2]))
# history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Predicción
# pred = model.predict(X)
