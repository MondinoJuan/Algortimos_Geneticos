import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class CropPredictionLSTM:
    def __init__(self, sequence_length=18, lstm_units=128, dropout_rate=0.2):
        """
        Inicializa el modelo LSTM para predicción de cosechas
        
        Args:
            sequence_length (int): Número de meses de datos climáticos a considerar (18 meses)
            lstm_units (int): Número de unidades LSTM
            dropout_rate (float): Tasa de dropout para regularización
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_climate = StandardScaler()
        self.scaler_soil = StandardScaler()
        self.scaler_target = StandardScaler()
        self.label_encoder_seeds = LabelEncoder()
        
    def prepare_climate_sequences(self, climate_data, target_dates):
        """
        Prepara las secuencias temporales de datos climáticos
        
        Args:
            climate_data (DataFrame): Datos climáticos con columnas fecha, temperatura_media_C, 
                                    humedad_relativa_%, velocidad_viento_m_s, precipitacion_mm_mes
            target_dates (list): Lista de fechas objetivo para las cuales obtener secuencias
            
        Returns:
            numpy.array: Array 3D con secuencias climáticas (samples, timesteps, features)
        """
        climate_features = ['temperatura_media_C', 'humedad_relativa_%', 
                          'velocidad_viento_m_s', 'precipitacion_mm_mes']
        
        sequences = []
        valid_indices = []
        
        # Convertir fecha a datetime si no lo está
        climate_data['fecha'] = pd.to_datetime(climate_data['fecha'])
        climate_data = climate_data.sort_values('fecha')
        
        for i, target_date in enumerate(target_dates):
            target_date = pd.to_datetime(target_date)
            # Obtener los 18 meses anteriores a la fecha objetivo
            start_date = target_date - pd.DateOffset(months=self.sequence_length)
            
            # Filtrar datos climáticos para el período
            period_data = climate_data[
                (climate_data['fecha'] >= start_date) & 
                (climate_data['fecha'] < target_date)
            ][climate_features]
            
            # Verificar que tenemos suficientes datos
            if len(period_data) >= self.sequence_length:
                # Si tenemos más datos, tomar los últimos 18 meses
                if len(period_data) > self.sequence_length:
                    period_data = period_data.tail(self.sequence_length)
                
                sequences.append(period_data.values)
                valid_indices.append(i)
            
        return np.array(sequences), valid_indices
    
    def prepare_training_data(self, climate_data, soil_data, crop_data, area_m2):
        """
        Prepara todos los datos para entrenamiento
        
        Args:
            climate_data (DataFrame): Datos climáticos históricos
            soil_data (DataFrame): Datos de suelo
            crop_data (DataFrame): Datos de cosechas históricas
            area_m2 (float): Área del campo en metros cuadrados
            
        Returns:
            tuple: (X_climate, X_soil, X_seeds, X_area, y) datos preparados para entrenamiento
        """
        # Preparar datos de cultivos
        crop_data['fecha_siembra'] = pd.to_datetime(crop_data['anio'].astype(str) + '-01-01')
        
        # Preparar secuencias climáticas
        X_climate, valid_indices = self.prepare_climate_sequences(
            climate_data, crop_data['fecha_siembra'].tolist()
        )
        
        # Filtrar datos de cultivos para índices válidos
        crop_data_valid = crop_data.iloc[valid_indices].reset_index(drop=True)
        
        # Preparar características del suelo (promedio de las coordenadas más cercanas)
        soil_features = ['bulk_density', 'ca_co3', 'coarse_fragments', 'ecec', 
                        'conductivity', 'organic_carbon', 'ph', 'clay', 'silt', 
                        'sand', 'water_retention']
        
        # Asumir que soil_data ya está filtrado por departamento/coordenadas más cercanas
        X_soil = np.tile(soil_data[soil_features].mean().values, (len(crop_data_valid), 1))
        
        # Preparar datos de semillas (asumiendo una columna 'tipo_semilla' en crop_data)
        # Si tienes múltiples semillas por registro, ajusta esta parte
        if 'tipo_semilla' not in crop_data_valid.columns:
            # Crear una columna de tipo de semilla basada en algún criterio
            crop_data_valid['tipo_semilla'] = 'default'
        
        X_seeds = self.label_encoder_seeds.fit_transform(crop_data_valid['tipo_semilla'])
        X_seeds = tf.keras.utils.to_categorical(X_seeds)
        
        # Área del campo (repetir para todos los ejemplos)
        X_area = np.full((len(crop_data_valid), 1), area_m2)
        
        # Variable objetivo: producción en kg (convertir de toneladas)
        y = crop_data_valid['produccion_tm'].values * 1000  # Convertir tm a kg
        
        return X_climate, X_soil, X_seeds, X_area, y, crop_data_valid
    
    def build_model(self, soil_features_dim, seed_categories_dim):
        """
        Construye la arquitectura del modelo LSTM
        
        Args:
            soil_features_dim (int): Dimensión de características del suelo
            seed_categories_dim (int): Número de categorías de semillas
        """
        # Input para secuencias climáticas
        climate_input = Input(shape=(self.sequence_length, 4), name='climate_sequence')
        
        # LSTM para procesar secuencias climáticas
        lstm_out = LSTM(self.lstm_units, return_sequences=True, 
                       dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(climate_input)
        lstm_out = LSTM(self.lstm_units//2, dropout=self.dropout_rate, 
                       recurrent_dropout=self.dropout_rate)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Input para características del suelo
        soil_input = Input(shape=(soil_features_dim,), name='soil_features')
        soil_dense = Dense(64, activation='relu')(soil_input)
        soil_dense = Dropout(self.dropout_rate)(soil_dense)
        soil_dense = Dense(32, activation='relu')(soil_dense)
        
        # Input para tipo de semillas
        seed_input = Input(shape=(seed_categories_dim,), name='seed_type')
        seed_dense = Dense(32, activation='relu')(seed_input)
        seed_dense = Dropout(self.dropout_rate)(seed_dense)
        
        # Input para área del campo
        area_input = Input(shape=(1,), name='field_area')
        area_dense = Dense(16, activation='relu')(area_input)
        
        # Concatenar todas las características
        combined = Concatenate()([lstm_out, soil_dense, seed_dense, area_dense])
        
        # Capas densas finales
        x = Dense(128, activation='relu')(combined)
        x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Salida: producción en kg
        output = Dense(1, activation='linear', name='production_kg')(x)
        
        # Crear el modelo
        self.model = Model(
            inputs=[climate_input, soil_input, seed_input, area_input],
            outputs=output
        )
        
        # Compilar el modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
    def train(self, X_climate, X_soil, X_seeds, X_area, y, validation_split=0.2, 
              epochs=100, batch_size=32, verbose=1):
        """
        Entrena el modelo LSTM
        
        Args:
            X_climate: Secuencias climáticas
            X_soil: Características del suelo
            X_seeds: Tipos de semillas (one-hot encoded)
            X_area: Área del campo
            y: Variable objetivo (producción en kg)
            validation_split: Porcentaje para validación
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            verbose: Nivel de verbosidad
        """
        # Normalizar datos
        X_climate_scaled = self.scaler_climate.fit_transform(
            X_climate.reshape(-1, X_climate.shape[-1])
        ).reshape(X_climate.shape)
        
        X_soil_scaled = self.scaler_soil.fit_transform(X_soil)
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Construir modelo si no existe
        if self.model is None:
            self.build_model(X_soil.shape[1], X_seeds.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
        ]
        
        # Entrenar el modelo
        history = self.model.fit(
            [X_climate_scaled, X_soil_scaled, X_seeds, X_area],
            y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X_climate, X_soil, X_seeds, X_area):
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            X_climate: Secuencias climáticas
            X_soil: Características del suelo
            X_seeds: Tipos de semillas
            X_area: Área del campo
            
        Returns:
            numpy.array: Predicciones de producción en kg
        """
        # Normalizar datos de entrada
        X_climate_scaled = self.scaler_climate.transform(
            X_climate.reshape(-1, X_climate.shape[-1])
        ).reshape(X_climate.shape)
        
        X_soil_scaled = self.scaler_soil.transform(X_soil)
        
        # Realizar predicción
        y_pred_scaled = self.model.predict([X_climate_scaled, X_soil_scaled, X_seeds, X_area])
        
        # Desnormalizar predicciones
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled).ravel()
        
        return y_pred
    
    def evaluate_model(self, X_climate, X_soil, X_seeds, X_area, y_true):
        """
        Evalúa el modelo comparando predicciones con valores reales
        
        Args:
            X_climate, X_soil, X_seeds, X_area: Datos de entrada
            y_true: Valores reales de producción
            
        Returns:
            dict: Métricas de evaluación
        """
        y_pred = self.predict(X_climate, X_soil, X_seeds, X_area)
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
    
    def save_model(self, filepath):
        """Guarda el modelo y los scalers"""
        self.model.save(f"{filepath}_model.h5")
        joblib.dump(self.scaler_climate, f"{filepath}_scaler_climate.pkl")
        joblib.dump(self.scaler_soil, f"{filepath}_scaler_soil.pkl")
        joblib.dump(self.scaler_target, f"{filepath}_scaler_target.pkl")
        joblib.dump(self.label_encoder_seeds, f"{filepath}_label_encoder_seeds.pkl")
    
    def load_model(self, filepath):
        """Carga el modelo y los scalers"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        self.scaler_climate = joblib.load(f"{filepath}_scaler_climate.pkl")
        self.scaler_soil = joblib.load(f"{filepath}_scaler_soil.pkl")
        self.scaler_target = joblib.load(f"{filepath}_scaler_target.pkl")
        self.label_encoder_seeds = joblib.load(f"{filepath}_label_encoder_seeds.pkl")

# Ejemplo de uso
def ejemplo_uso():
    """
    Ejemplo de cómo usar el modelo LSTM para predicción de cosechas
    """
    # Crear instancia del modelo
    lstm_model = CropPredictionLSTM(sequence_length=18, lstm_units=128)
    
    # Cargar datos (aquí necesitarías cargar tus DataFrames reales)
    # climate_data = pd.read_csv('datos_climaticos.csv')
    # soil_data = pd.read_csv('datos_suelo.csv')
    # crop_data = pd.read_csv('datos_cosechas.csv')
    # area_m2 = 100000  # Ejemplo: 10 hectáreas
    
    # Preparar datos de entrenamiento
    # X_climate, X_soil, X_seeds, X_area, y, crop_info = lstm_model.prepare_training_data(
    #     climate_data, soil_data, crop_data, area_m2
    # )
    
    # Entrenar el modelo
    # history = lstm_model.train(X_climate, X_soil, X_seeds, X_area, y, epochs=100)
    
    # Evaluar el modelo
    # metrics = lstm_model.evaluate_model(X_climate, X_soil, X_seeds, X_area, y)
    # print("Métricas del modelo:", metrics)
    
    # Hacer predicciones para nuevas combinaciones
    # predictions = lstm_model.predict(X_climate_new, X_soil_new, X_seeds_new, X_area_new)
    
    # Guardar el modelo
    # lstm_model.save_model('modelo_cosechas_argentina')
    
    print("Modelo LSTM listo para usar en predicción de cosechas")

if __name__ == "__main__":
    ejemplo_uso()