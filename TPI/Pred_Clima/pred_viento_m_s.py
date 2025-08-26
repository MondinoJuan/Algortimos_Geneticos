import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Parámetros globales
WINDOW_SIZE = 30
EPOCHS = 200
BATCH_SIZE = 32

scaler = StandardScaler()

def load_and_preprocess(csv_path, col_name):
    """Carga datos de viento y los normaliza"""
    data = pd.read_csv(csv_path, parse_dates=["fecha"])
    data = data[["fecha", col_name]].copy()
    data[col_name] = scaler.fit_transform(data[[col_name]])
    return data

def create_sequences(data, col_name, window_size):
    """Crea ventanas deslizantes para entrenamiento"""
    X, y = [], []
    values = data[col_name].values
    for i in range(len(values) - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size])
    return np.array(X), np.array(y)

def build_model(window_size):
    """Modelo LSTM mejorado con Dropout"""
    model = Sequential([
        InputLayer((window_size, 1)),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def plot_results(dates, true, pred, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true, "ko-", label="True Values")
    plt.plot(dates, pred, "ro-", label="Predictions")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def predecir_proximos_meses(model, data, col_name, window_size=30, steps=14):
    """Predice próximos `steps` meses usando la última ventana"""
    ultimos = data[col_name].values[-window_size:]
    predicciones = []

    entrada = ultimos.reshape(1, window_size, 1)
    for _ in range(steps):
        pred = model.predict(entrada, verbose=0)[0,0]
        predicciones.append(pred)
        ultimos = np.append(ultimos[1:], pred)
        entrada = ultimos.reshape(1, window_size, 1)

    # Invertir la normalización
    predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1,1)).flatten()
    return predicciones

def main(csv_path, col_name="velocidad_viento_m_s"):
    # 1. Cargar y normalizar
    data = load_and_preprocess(csv_path, col_name)

    # 2. Crear secuencias
    X, y = create_sequences(data, col_name, WINDOW_SIZE)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 3. Train / Test split
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = data["fecha"][WINDOW_SIZE:split+WINDOW_SIZE], data["fecha"][split+WINDOW_SIZE:]

    # 4. Modelo
    model_path = f"model/model_{col_name}.keras"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model(WINDOW_SIZE)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[checkpoint, reduce_lr], verbose=1)

    # 5. Predicciones
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # Invertir normalización
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    train_pred_inv = scaler.inverse_transform(train_pred).flatten()
    test_pred_inv = scaler.inverse_transform(test_pred).flatten()

    # 6. Graficar
    plot_results(dates_train, y_train_inv, train_pred_inv, "Train Data", f"Velocidad viento ({col_name})")
    plot_results(dates_test, y_test_inv, test_pred_inv, "Test Data", f"Velocidad viento ({col_name})")

    # 7. Métricas
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)))
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test_inv, test_pred_inv)))

    # 8. Predicción futura
    pred_futuro = predecir_proximos_meses(model, data, col_name, WINDOW_SIZE, steps=14)
    print("Predicciones próximas 14 meses:", pred_futuro)

if __name__ == "__main__":
    main("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv", col_name="velocidad_viento_m_s")
