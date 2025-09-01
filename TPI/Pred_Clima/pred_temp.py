import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
from keras.models import Sequential
from keras.losses import MeanSquaredError 
from keras.metrics import RootMeanSquaredError
from keras.layers import Dense, InputLayer, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from Recuperacion_de_datos.Clima.Recupero_clima_NASA_Mensual import main as creo_archivo_clima

WINDOW_SIZE = 14

# Graficas
def plot_predictions(pred, y_true, label, period) :
    # Convert period to datetime if it's not already
    period = pd.to_datetime(period)

    plt.figure(figsize=(10, 6))
    plt.plot(period, y_true, label='True Values', marker='o', color='black')
    plt.plot(period, pred, label='Predictions', marker='o', color='red')

    plt.xlabel('Date')
    plt.ylabel('Mean temperature' )
    plt.title(label)

    plt.legend()

    if label == "Train Data":
        plt.gca().xaxis.set_major_locator(YearLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    elif label == "Test Data":
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y' ) )

    # Rotate x-axis labels for better visibility
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt. show()

# Build and train the NN
def train_neural_network(X, y, epochs=100, learning_rate=0.001):
    model = Sequential([
        InputLayer((WINDOW_SIZE, 1)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, 'linear')
    ])
    
    checkpoint = ModelCheckpoint("model/model_temp.keras", save_best_only=True, monitor='val_loss')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.fit(X, y, epochs=epochs, callbacks=[checkpoint])

def data_to_input_and_out(data):
    X, y = [], []
    for i in range(len(data) - WINDOW_SIZE):
        X.append(data['temperatura_media_C'][i:i + WINDOW_SIZE])
        y.append(data['temperatura_media_C'][i + WINDOW_SIZE])
    return np.array(X), np.array(y)

def train_LSTM_temp(data):
    # Load data
    split_idx = int(len(data) * 0.85)
    train = data.iloc[:split_idx].reset_index(drop=True)
    test = data.iloc[split_idx:].reset_index(drop=True)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # Preprocessing
    train_input, train_output = data_to_input_and_out(train)
    test_input, test_output = data_to_input_and_out(test)

    train_neural_network(train_input, train_output)


    # Predictions
    model = load_model("model/model_temp.keras")

    train_pred = model.predict(train_input)
    test_pred = model.predict(test_input)

    train_period = train[0:len(train) - WINDOW_SIZE]['fecha']
    test_period = test[0:len(test) - WINDOW_SIZE]['fecha']

    plot_predictions(train_pred, train_output, "Train Data", train_period)
    plot_predictions(test_pred, test_output, "Test Data", test_period)

def predecir_proximos_meses(latitud, longitud, WINDOW_SIZE=14, steps=14):
    """
    Genera predicciones para los próximos `steps` meses
    usando la última ventana de `WINDOW_SIZE` observaciones.
    """
    data = creo_archivo_clima(latitud, longitud)
    ultimos = data['temperatura_media_C'].values[-WINDOW_SIZE:]
    predicciones = []
    model = load_model("model/model_temp.keras")

    entrada = ultimos.reshape(1, WINDOW_SIZE, 1)
    for _ in range(steps):
        pred = model.predict(entrada, verbose=0)[0,0]
        predicciones.append(pred)

        # desplazar ventana: sacar el primero y meter el predicho al final
        ultimos = np.append(ultimos[1:], pred)
        entrada = ultimos.reshape(1, WINDOW_SIZE, 1)

    return np.array(predicciones)

if __name__ == "__main__":
    # Example usage
    predicciones_futuras = predecir_proximos_meses(-34.6037, -58.3816, WINDOW_SIZE=WINDOW_SIZE, steps=14)
    print("Predicciones próximas 14 meses:", predicciones_futuras)
