import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class GRUSeedPredictor:
    def __init__(self, sequence_length=12, n_features=10, n_seed_types=20):
        """
        Inicializa el predictor de semillas con GRU
        
        Args:
            sequence_length: Número de meses/períodos históricos a considerar
            n_features: Número de características por período (clima, suelo, etc.)
            n_seed_types: Número de tipos de semillas diferentes
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_seed_types = n_seed_types
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def build_model(self):
        """
        Construye la arquitectura de la red neuronal GRU
        """
        model = Sequential([
            # Primera capa GRU con return_sequences=True para apilar capas
            GRU(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Segunda capa GRU
            GRU(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Tercera capa GRU (última capa recurrente)
            GRU(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            # Capas densas para la clasificación final
            Dense(64, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Capa de salida para clasificación de semillas
            Dense(self.n_seed_types, activation='softmax')
        ])
        
        # Compilar el modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X_raw, y_raw, location_data):
        """
        Prepara los datos para entrenamiento
        
        Args:
            X_raw: Datos históricos climáticos y de suelo (shape: [samples, sequence_length, features])
            y_raw: Etiquetas de semillas exitosas
            location_data: Datos de ubicación (lat, lon, metros_cuadrados, tipo_suelo)
        """
        # Normalizar datos temporales
        X_reshaped = X_raw.reshape(-1, self.n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X_raw.shape)
        
        # Combinar datos temporales con datos de ubicación
        # Expandir location_data para cada timestep
        location_expanded = np.repeat(location_data[:, np.newaxis, :], self.sequence_length, axis=1)
        X_combined = np.concatenate([X_scaled, location_expanded], axis=2)
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y_raw)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=self.n_seed_types)
        
        return X_combined, y_categorical
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Entrena el modelo GRU
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_best_seed(self, climate_sequence, location_data):
        """
        Predice la mejor semilla para una secuencia climática y ubicación dada
        
        Args:
            climate_sequence: Secuencia de datos climáticos (shape: [sequence_length, features])
            location_data: Datos de ubicación [lat, lon, metros_cuadrados, tipo_suelo]
        
        Returns:
            Tuple: (mejor_semilla, probabilidad, top_3_semillas)
        """
        # Preparar datos de entrada
        X_input = climate_sequence.reshape(1, self.sequence_length, -1)
        
        # Normalizar datos climáticos
        X_climate_scaled = self.scaler.transform(X_input.reshape(-1, self.n_features))
        X_climate_scaled = X_climate_scaled.reshape(1, self.sequence_length, self.n_features)
        
        # Añadir datos de ubicación
        location_expanded = np.repeat(location_data.reshape(1, 1, -1), self.sequence_length, axis=1)
        X_combined = np.concatenate([X_climate_scaled, location_expanded], axis=2)
        
        # Realizar predicción
        predictions = self.model.predict(X_combined)[0]
        
        # Obtener resultados
        best_seed_idx = np.argmax(predictions)
        best_seed_name = self.label_encoder.inverse_transform([best_seed_idx])[0]
        confidence = predictions[best_seed_idx]
        
        # Top 3 semillas
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_seeds = [
            (self.label_encoder.inverse_transform([idx])[0], predictions[idx])
            for idx in top_3_indices
        ]
        
        return best_seed_name, confidence, top_3_seeds
    
    def fitness_function(self, climate_sequence, location_data, target_seed):
        """
        Función de fitness para el algoritmo genético
        
        Args:
            climate_sequence: Secuencia de datos climáticos
            location_data: Datos de ubicación
            target_seed: Semilla objetivo para evaluar
        
        Returns:
            float: Valor de fitness (probabilidad de éxito)
        """
        # Preparar datos
        X_input = climate_sequence.reshape(1, self.sequence_length, -1)
        X_climate_scaled = self.scaler.transform(X_input.reshape(-1, self.n_features))
        X_climate_scaled = X_climate_scaled.reshape(1, self.sequence_length, self.n_features)
        
        location_expanded = np.repeat(location_data.reshape(1, 1, -1), self.sequence_length, axis=1)
        X_combined = np.concatenate([X_climate_scaled, location_expanded], axis=2)
        
        # Obtener predicción
        predictions = self.model.predict(X_combined)[0]
        
        # Obtener índice de la semilla objetivo
        target_seed_encoded = self.label_encoder.transform([target_seed])[0]
        
        # Retornar probabilidad de éxito para esa semilla
        return predictions[target_seed_encoded]
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo en datos de prueba
        """
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métricas de evaluación
        accuracy = np.mean(y_pred == y_true)
        
        print(f"Precisión del modelo: {accuracy:.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(y_true, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return accuracy

# Ejemplo de uso del modelo
def example_usage():
    """
    Ejemplo de cómo usar el modelo GRU para predicción de semillas
    """
    # Parámetros del modelo
    sequence_length = 12  # 12 meses de datos históricos
    n_features = 8  # Características climáticas por mes
    n_seed_types = 15  # Tipos de semillas disponibles
    
    # Crear instancia del predictor
    predictor = GRUSeedPredictor(sequence_length, n_features, n_seed_types)
    
    # Construir modelo
    model = predictor.build_model()
    print("Arquitectura del modelo:")
    model.summary()
    
    # Simular datos de ejemplo (en tu caso, estos vendrían de las bases de datos)
    n_samples = 1000
    
    # Datos climáticos históricos
    # Formato: [muestra, mes, características]
    # Características: temp_max, temp_min, precipitacion, humedad, radiacion_solar, etc.
    X_climate = np.random.rand(n_samples, sequence_length, n_features)
    
    # Datos de ubicación
    # Formato: [latitud, longitud, metros_cuadrados, tipo_suelo_codificado]
    location_data = np.random.rand(n_samples, 4)
    
    # Etiquetas de semillas exitosas
    seed_types = ['Soja', 'Maiz', 'Trigo', 'Girasol', 'Cebada', 'Avena', 
                  'Sorgo', 'Mijo', 'Quinoa', 'Amaranto', 'Centeno', 
                  'Triticale', 'Colza', 'Lino', 'Cartamo']
    y_seeds = np.random.choice(seed_types, n_samples)
    
    # Preparar datos
    X_combined, y_categorical = predictor.prepare_data(X_climate, y_seeds, location_data)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_categorical, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\nForma de datos de entrenamiento: {X_train.shape}")
    print(f"Forma de etiquetas: {y_train.shape}")
    
    # Entrenar modelo
    print("\nEntrenando modelo...")
    history = predictor.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    accuracy = predictor.evaluate_model(X_test, y_test)
    
    # Ejemplo de predicción
    print("\nEjemplo de predicción:")
    sample_climate = X_climate[0]  # Primer ejemplo
    sample_location = location_data[0]  # Primera ubicación
    
    best_seed, confidence, top_3 = predictor.predict_best_seed(
        sample_climate, sample_location
    )
    
    print(f"Mejor semilla recomendada: {best_seed}")
    print(f"Confianza: {confidence:.4f}")
    print("Top 3 semillas:")
    for i, (seed, prob) in enumerate(top_3, 1):
        print(f"  {i}. {seed}: {prob:.4f}")
    
    # Ejemplo de función de fitness
    print("\nEjemplo de función de fitness:")
    fitness_soja = predictor.fitness_function(sample_climate, sample_location, 'Soja')
    fitness_maiz = predictor.fitness_function(sample_climate, sample_location, 'Maiz')
    
    print(f"Fitness para Soja: {fitness_soja:.4f}")
    print(f"Fitness para Maíz: {fitness_maiz:.4f}")

if __name__ == "__main__":
    example_usage()