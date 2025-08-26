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
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("Recuperacion_de_datos/Clima/clima_nasa_mensual_44_anios.csv")
split_idx = int(len(data) * 0.85)
train = data.iloc[:split_idx].reset_index(drop=True)
test = data.iloc[split_idx:].reset_index(drop=True)

features = ['temperatura_media_C','humedad_relativa_%','velocidad_viento_m_s','velocidad_viento_km_h','precipitacion_mm_mes']

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preprocessing

window_size = 14



def data_to_input_and_out(data):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i + window_size].values)
        y.append(data.iloc[i + window_size][features].values)

    return np.array(X), np.array(y)

train_input, train_output = data_to_input_and_out(train)
test_input, test_output = data_to_input_and_out(test)

train_input = train_input.astype('float32')
train_output = train_output.astype('float32')
test_input = test_input.astype('float32')
test_output = test_output.astype('float32')

# Build and train the NN
def train_neural_network(X, y, epochs=100, learning_rate=0.001):
    n_features = len(features)
    
    model = Sequential([
        InputLayer((window_size, n_features)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(n_features, 'linear')
    ])
    
    checkpoint = ModelCheckpoint("model/model.keras", save_best_only=True, monitor='val_loss')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.fit(X, y, epochs=epochs, callbacks=[checkpoint])

train_neural_network(train_input, train_output)


# Predictions
model = load_model("model/model.keras")

train_pred = model.predict(train_input)
test_pred = model.predict(test_input)

"""
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
"""
#train_period = train[0:len(train) - window_size]['fecha']
#test_period = test[0:len(test) - window_size]['fecha']

#plot_predictions(train_pred, train_output, "Train Data", train_period)
#plot_predictions(test_pred, test_output, "Test Data", test_period)


def plot_feature(idx, feature_name, pred, y_true, period, label):
    plt.figure(figsize=(10,6))
    plt.plot(period, y_true[:, idx], label="True", marker="o", color="black")
    plt.plot(period, pred[:, idx], label="Pred", marker="o", color="red")
    plt.title(f"{label} - {feature_name}")
    plt.xlabel("Date")
    plt.ylabel(feature_name)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

train_period = pd.to_datetime(train.iloc[:len(train)-window_size]['fecha'])
test_period  = pd.to_datetime(test.iloc[:len(test)-window_size]['fecha'])

for i, feat in enumerate(features):
    plot_feature(i, feat, train_pred, train_output, train_period, "Train Data")
    plot_feature(i, feat, test_pred,  test_output,  test_period,  "Test Data")

for i in range(len(train_pred)):
    print(f"Mes {i+1}:")
    for j, feat in enumerate(features):
        print(f"  {feat}: {train_pred[i, j]}")
    print("-------------------")

