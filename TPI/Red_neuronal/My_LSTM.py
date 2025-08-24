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
from sklearn.preprocessing import OneHotEncoder


# Load data
data = pd.read_csv("df_con_prod.csv")
split_idx = int(len(data) * 0.85)

# ------------------------------------------------------
# Codifico las columnas categoricas
categorical_cols = ['cultivo_nombre', 'departamento_nombre']

# Crear encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[categorical_cols])

# Poner nombres a las nuevas columnas
encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)

# Reemplazar categóricas por one-hot
data = pd.concat([data.drop(columns=categorical_cols), encoded_df], axis=1)

# Asegurar que todas las columnas sean numéricas
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

print("Tipos de datos finales:\n", data.dtypes)

# ------------------------------------------------------


train = data.iloc[:split_idx].reset_index(drop=True)
test = data.iloc[split_idx:].reset_index(drop=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Preprocessing

window_size = 8
'''
def data_to_input_and_out(data):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[['cultivo_nombre', 'anio', 'departamento_nombre', 'organic_carbon', 'ph', 'clay', 'silt', 'sand', 'temperatura_media_C', 
                       'humedad_relativa_%', 'velocidad_viento_m_s', 'velocidad_viento_km_h', 'precipitacion_mm_mes', 'superficie_sembrada_ha']].iloc[i:i + window_size].values)
        y.append(data['produccion_tn'][i + window_size])
    return np.array(X), np.array(y)
'''

def data_to_input_and_out(data):
    X, y = [], []
    feature_cols = [c for c in data.columns if c != 'produccion_tn']  # todas menos la salida
    for i in range(len(data) - window_size):
        X.append(data[feature_cols].iloc[i:i + window_size].values)
        y.append(data['produccion_tn'].iloc[i + window_size])
    return np.array(X), np.array(y)

train_input, train_output = data_to_input_and_out(train)
test_input, test_output = data_to_input_and_out(test)

# Build and train the NN
'''
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
'''

def train_neural_network(X, y, epochs=100, learning_rate=0.001):
    n_features = X.shape[2]  # cantidad de columnas
    model = Sequential([
        InputLayer((window_size, n_features)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, 'linear')
    ])
    
    checkpoint = ModelCheckpoint("model/model.keras", save_best_only=True, monitor='val_loss')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.fit(X, y, epochs=epochs, validation_split=0.1, callbacks=[checkpoint])

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
    plt.ylabel('Produccion' )
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


train_period = train[0:len(train) - window_size]['anio']
test_period = test[0:len(test) - window_size]['anio']

plot_predictions(train_pred, train_output, "Train Data", train_period)
plot_predictions(test_pred, test_output, "Test Data", test_period)