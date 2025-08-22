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


# Load data
data = pd.read_csv("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv")
split_idx = int(len(data) * 0.85)
train = data.iloc[:split_idx].reset_index(drop=True)
test = data.iloc[split_idx:].reset_index(drop=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preprocessing

window_size = 14

def data_to_input_and_out(data):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data['temperatura_media_C'][i:i + window_size])
        y.append(data['temperatura_media_C'][i + window_size])
    return np.array(X), np.array(y)

train_input, train_output = data_to_input_and_out(train)
test_input, test_output = data_to_input_and_out(test)

# Build and train the NN
def train_neural_network(X, y, epochs=100, learning_rate=0.001):
    model = Sequential([
        InputLayer((window_size, 1)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, 'linear')
    ])
    
    checkpoint = ModelCheckpoint("model/model.keras", save_best_only=True, monitor='val_loss')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.fit(X, y, epochs=epochs, callbacks=[checkpoint])

train_neural_network(train_input, train_output)


# Predictions
model = load_model("model/model.keras")

train_pred = model.predict(train_input)
test_pred = model.predict(test_input)


# Graficas
def plot_predictions(pred, y_true, label, period) :
    # Convert period to datetime if it's not already
    period = pd.to_datetime(period)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(period, y_true, label='True Values', marker='o', color='black')
    plt.plot(period, pred, label='Predictions', marker='o', color='red')

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Mean temperature' )
    plt.title(label)

    # Adding legend
    plt.legend()

    if label == "Train Data":
        # Format X-axis to show only years
        plt.gca().xaxis.set_major_locator(YearLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    elif label == "Test Data":
        # Format X-axis to show only months
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y' ) )

    # Rotate x-axis labels for better visibility
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt. show()


train_period = train[0:len(train) - window_size]['fecha']
test_period = test[0:len(test) - window_size]['fecha']

plot_predictions(train_pred, train_output, "Train Data", train_period)
plot_predictions(test_pred, test_output, "Test Data", test_period)