import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GRU, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping

# Supongamos que tenemos:
# - Datos temporales (clima históricos): secuencia de 12 meses (shape=[None, 12, num_features_temporales])
# - Datos estáticos (suelo, ubicación): vector de características (shape=[None, num_features_estaticas])

# Hiperparámetros
num_classes = 10  # Número de semillas a predecir
gru_units = 64
dense_units = 128
dropout_rate = 0.3
learning_rate = 0.001

# Capas de entrada
input_temporal = Input(shape=(12, 5), name='input_temporal')  # Ej: 12 meses, 5 features (temp, lluvia, etc.)
input_estatico = Input(shape=(7,), name='input_estatico')      # Ej: 7 features de suelo/ubicación

# Parte GRU para datos temporales
x_temporal = GRU(gru_units, return_sequences=False)(input_temporal)
x_temporal = BatchNormalization()(x_temporal)
x_temporal = Dropout(dropout_rate)(x_temporal)

# Parte Densa para datos estáticos
x_estatico = Dense(dense_units, activation='relu')(input_estatico)
x_estatico = BatchNormalization()(x_estatico)
x_estatico = Dropout(dropout_rate)(x_estatico)

# Concatenar ambas partes
x = Concatenate()([x_temporal, x_estatico])

# Capas densas finales
x = Dense(dense_units, activation='relu')(x)
x = Dropout(dropout_rate)(x)
output = Dense(num_classes, activation='softmax')(x)  # Clasificación multiclase

# Modelo
model = Model(inputs=[input_temporal, input_estatico], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Resumen
model.summary()

# Entrenamiento (ejemplo)
# history = model.fit(
#     x=[X_temporal_train, X_estatico_train],
#     y=y_train_onehot,
#     validation_data=([X_temporal_val, X_estatico_val], y_val_onehot),
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stopping]
# )