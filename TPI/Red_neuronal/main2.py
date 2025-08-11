# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ------------------------------------------------------------
# Modelo LSTM para producción (kg)
# ------------------------------------------------------------
def build_lstm_production_model(
    seq_len: int,
    n_climate_features: int,
    n_soil_features: int,
    n_seeds: int,
    lstm_units=(64, 32),
    dense_units=(64, 32),
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Crea un modelo LSTM multimodal que predice producción total (kg)
    a partir de:
      - clima_seq: (batch, seq_len, n_climate_features)
      - suelo:     (batch, n_soil_features)
      - area_m2:   (batch, 1)
      - semillas_mix: (batch, n_seeds) vector de proporciones o pesos

    Devuelve:
      - model: keras.Model compilado con pérdida MSE y métricas MAE, RMSE, MAPE
    """

    # -------- Rama clima (secuencial) --------
    clima_in = keras.Input(shape=(seq_len, n_climate_features), name="clima_seq")

    # Normalización por característica (opcional, simple). Si prefieres Normalization.adapt(),
    # normaliza antes y elimina esta capa.
    x = layers.LayerNormalization(axis=-1, name="clima_layernorm")(clima_in)

    # LSTM apilado
    x = layers.LSTM(lstm_units[0], return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.LSTM(lstm_units[1], return_sequences=False, name="lstm_2")(x)
    clima_repr = layers.Dense(32, activation="relu", name="clima_dense")(x)

    # -------- Rama suelo (estático) --------
    suelo_in = keras.Input(shape=(n_soil_features,), name="suelo")
    s = layers.LayerNormalization(axis=-1, name="suelo_layernorm")(suelo_in)
    s = layers.Dense(64, activation="relu", name="suelo_dense1")(s)
    suelo_repr = layers.Dense(32, activation="relu", name="suelo_dense2")(s)

    # -------- Rama área (m² → ha) --------
    area_in = keras.Input(shape=(1,), name="area_m2")
    area_ha = layers.Lambda(lambda t: t / 10000.0, name="to_hectareas")(area_in)
    a = layers.Dense(16, activation="relu", name="area_dense1")(area_ha)
    area_repr = layers.Dense(8, activation="relu", name="area_dense2")(a)

    # -------- Rama semillas (combinación) --------
    semillas_in = keras.Input(shape=(n_seeds,), name="semillas_mix")
    # Normaliza a simplex (suma 1) para estabilizar combinaciones
    semillas_norm = layers.Lambda(
        lambda t: t / (tf.reduce_sum(t, axis=-1, keepdims=True) + 1e-8),
        name="semillas_normalize"
    )(semillas_in)
    m = layers.Dense(64, activation="relu", name="semillas_dense1")(semillas_norm)
    semillas_repr = layers.Dense(32, activation="relu", name="semillas_dense2")(m)

    # -------- Fusión y cabeza de predicción --------
    h = layers.Concatenate(name="fusion")([clima_repr, suelo_repr, area_repr, semillas_repr])
    h = layers.Dense(dense_units[0], activation="relu", name="head_dense1")(h)
    h = layers.Dropout(dropout_rate, name="dropout_head")(h)
    h = layers.Dense(dense_units[1], activation="relu", name="head_dense2")(h)

    # Salida en kg, no negativa
    out = layers.Dense(1, activation="relu", name="produccion_kg")(h)

    model = keras.Model(
        inputs=[clima_in, suelo_in, area_in, semillas_in],
        outputs=out,
        name="LSTM_ProduccionKG"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MAE(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )
    return model


# ------------------------------------------------------------
# Ejemplo mínimo de uso
# ------------------------------------------------------------
def ejemplo_entrenamiento():
    """
    Ejemplo ilustrativo (sintético). Reemplaza con tus tensores reales:

      X_clima:        (N, 18, F_clima)
      X_suelo:        (N, F_suelo)
      X_area_m2:      (N, 1)
      X_semillas_mix: (N, S)
      y_kg:           (N, 1)

    Nota: prepara y alinea los datos acorde a tu lógica de negocio
    (18 meses previos a la siembra cuya cosecha produce y_kg).
    """
    import numpy as np

    # Dimensiones de ejemplo
    N = 1024
    seq_len = 18
    F_clima = 6   # ajusta a tus features climáticas
    F_suelo = 12  # ajusta a tus features de suelo
    S = 8         # número de semillas en tu espacio de combinaciones

    rng = np.random.default_rng(42)

    X_clima = rng.normal(size=(N, seq_len, F_clima)).astype("float32")
    X_suelo = rng.normal(size=(N, F_suelo)).astype("float32")
    X_area_m2 = rng.uniform(5e4, 5e6, size=(N, 1)).astype("float32")  # 5 ha a 500 ha
    # Mezclas de semillas aleatorias (no normalizadas); el modelo las normaliza
    X_semillas = rng.uniform(0, 1, size=(N, S)).astype("float32")

    # Etiquetas sintéticas en kg (reemplaza por produccion_tm*1000)
    y_kg = (rng.uniform(0.5, 1.5, size=(N, 1)) *
            (X_area_m2 / 10000.0) * 3000.0).astype("float32")  # ~ rendimiento * ha

    model = build_lstm_production_model(
        seq_len=seq_len,
        n_climate_features=F_clima,
        n_soil_features=F_suelo,
        n_seeds=S,
        lstm_units=(64, 32),
        dense_units=(64, 32),
        dropout_rate=0.2,
        learning_rate=1e-3,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_rmse"),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=1, monitor="val_rmse"),
    ]

    history = model.fit(
        {"clima_seq": X_clima, "suelo": X_suelo, "area_m2": X_area_m2, "semillas_mix": X_semillas},
        y_kg,
        validation_split=0.2,
        epochs=200,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    # Predicción para una nueva combinación de semillas
    y_pred = model.predict(
        {"clima_seq": X_clima[:5], "suelo": X_suelo[:5], "area_m2": X_area_m2[:5], "semillas_mix": X_semillas[:5]},
        verbose=0
    )
    print("Predicción (kg) primeras 5 muestras:", y_pred.squeeze()[:5])

    return model, history


if __name__ == "__main__":
    # Comentá esta línea si sólo vas a importar el modelo
    ejemplo_entrenamiento()