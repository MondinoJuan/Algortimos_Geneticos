import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
from keras.models import Sequential, load_model
from keras.losses import MeanSquaredError 
from keras.metrics import RootMeanSquaredError
from keras.layers import Dense, InputLayer, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import main as creo_archivo_clima

WINDOW_SIZE = 30
scaler = MinMaxScaler()

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

def plot_predictions(pred, y_true, label, period):
    period = pd.to_datetime(period)
    plt.figure(figsize=(10, 6))
    plt.plot(period, y_true, label='True Values', marker='o', color='black')
    plt.plot(period, pred, label='Predictions', marker='o', color='red')
    plt.xlabel('Date')
    plt.ylabel('Velocidad viento (km/h)')
    plt.title(label)
    plt.legend()
    if label == "Train Data":
        plt.gca().xaxis.set_major_locator(YearLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    elif label == "Test Data":
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.show()

def train_neural_network(X, y, epochs=100, learning_rate=0.001):
    model = Sequential([
        InputLayer((WINDOW_SIZE, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(16, activation='relu'),
        Dense(1, 'linear')
    ])
    checkpoint = ModelCheckpoint("model/model_v_km_h.keras", save_best_only=True, monitor='val_loss')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.fit(X, y, epochs=epochs, callbacks=[checkpoint])

def data_to_input_and_out(data):
    X, y = [], []
    for i in range(len(data) - WINDOW_SIZE):
        X.append(data['velocidad_viento_km_h'][i:i + WINDOW_SIZE])
        y.append(data['velocidad_viento_km_h'][i + WINDOW_SIZE])
    return np.array(X), np.array(y)

def train_LSTM_viento_kmh(data):
    # Normalizar
    data['velocidad_viento_km_h'] = scaler.fit_transform(data[['velocidad_viento_km_h']])
    split_idx = int(len(data) * 0.85)
    train = data.iloc[:split_idx].reset_index(drop=True)
    test = data.iloc[split_idx:].reset_index(drop=True)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    train_input, train_output = data_to_input_and_out(train)
    test_input, test_output = data_to_input_and_out(test)
    train_neural_network(train_input, train_output)
    model = load_model("model/model_v_km_h.keras")
    train_pred = model.predict(train_input)
    test_pred = model.predict(test_input)
    # Invertir escala
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    train_output = scaler.inverse_transform(train_output.reshape(-1,1))
    test_output = scaler.inverse_transform(test_output.reshape(-1,1))
    train_period = train[0:len(train) - WINDOW_SIZE]['fecha']
    test_period = test[0:len(test) - WINDOW_SIZE]['fecha']
    plot_predictions(train_pred, train_output, "Train Data", train_period)
    plot_predictions(test_pred, test_output, "Test Data", test_period)

def predecir_proximos_meses(latitud, longitud, WINDOW_SIZE=14, steps=14):
    data = creo_archivo_clima(latitud, longitud)
    ultimos = data['velocidad_viento_km_h'].values[-WINDOW_SIZE:]
    ultimos = scaler.fit_transform(ultimos.reshape(-1,1)).flatten()
    predicciones = []
    model = load_model("model/model_v_km_h.keras")
    entrada = ultimos.reshape(1, WINDOW_SIZE, 1)
    for _ in range(steps):
        pred = model.predict(entrada, verbose=0)[0,0]
        predicciones.append(pred)
        ultimos = np.append(ultimos[1:], pred)
        entrada = ultimos.reshape(1, WINDOW_SIZE, 1)
    predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1,1)).flatten()
    return predicciones

if __name__ == "__main__":
    predicciones_futuras = predecir_proximos_meses(-34.6037, -58.3816, WINDOW_SIZE=WINDOW_SIZE, steps=14)
    print("Predicciones viento km/h próximos 14 meses:", predicciones_futuras)
