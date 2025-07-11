import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuración para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

class AgriculturaGRU:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def generar_datos_simulados(self, n_samples=5000):
        """
        Genera datos simulados para agricultura argentina con distribución balanceada
        """
        # Definir zonas agrícolas argentinas
        zonas = ['Pampeana', 'NOA', 'NEA', 'Cuyo', 'Patagonia']
        
        # Tipos de semillas comunes en Argentina
        semillas = ['Soja', 'Maíz', 'Trigo', 'Girasol', 'Sorgo', 'Cebada']
        
        # Generar datos con distribución más balanceada
        data = []
        samples_per_zone = n_samples // len(zonas)
        
        for zona in zonas:
            for i in range(samples_per_zone):
                # Características de la zona
                metros_cuadrados = np.random.uniform(1000, 50000)  # Entre 1000 y 50000 m²
                
                # Características del suelo (pH, materia orgánica, nitrógeno, fósforo, potasio)
                ph_suelo = np.random.normal(6.5, 0.8)
                materia_organica = np.random.uniform(1.5, 4.5)
                nitrogeno = np.random.uniform(10, 50)
                fosforo = np.random.uniform(8, 40)
                potasio = np.random.uniform(100, 400)
                
                # Datos climáticos ajustados por zona
                temperatura_media = self._generar_temperatura_zona(zona)
                precipitacion = self._generar_precipitacion_zona(zona)
                humedad = np.random.uniform(40, 90, 12)
                
                # Precios históricos (secuencia temporal de 12 meses)
                precio_base = np.random.uniform(200, 800)  # USD por tonelada
                variacion_precio = np.random.normal(0, 0.1, 12)
                precios = precio_base * (1 + variacion_precio)
                
                # Seleccionar semilla óptima basada en lógica de negocio
                semilla_optima = self._determinar_semilla_optima(
                    zona, ph_suelo, temperatura_media, precipitacion, 
                    metros_cuadrados, precios
                )
                
                # Crear registro
                registro = {
                    'zona': zona,
                    'metros_cuadrados': metros_cuadrados,
                    'ph_suelo': ph_suelo,
                    'materia_organica': materia_organica,
                    'nitrogeno': nitrogeno,
                    'fosforo': fosforo,
                    'potasio': potasio,
                    'semilla_optima': semilla_optima
                }
                
                # Agregar datos temporales
                for mes in range(12):
                    registro[f'temp_mes_{mes+1}'] = temperatura_media[mes]
                    registro[f'precip_mes_{mes+1}'] = precipitacion[mes]
                    registro[f'humedad_mes_{mes+1}'] = humedad[mes]
                    registro[f'precio_mes_{mes+1}'] = precios[mes]
                
                data.append(registro)
        
        return pd.DataFrame(data)
    
    def _generar_temperatura_zona(self, zona):
        """Genera temperaturas realistas según la zona"""
        if zona == 'Pampeana':
            return np.random.normal(18, 6, 12)
        elif zona == 'NOA':
            return np.random.normal(22, 4, 12)
        elif zona == 'NEA':
            return np.random.normal(24, 3, 12)
        elif zona == 'Cuyo':
            return np.random.normal(16, 5, 12)
        else:  # Patagonia
            return np.random.normal(10, 4, 12)
    
    def _generar_precipitacion_zona(self, zona):
        """Genera precipitaciones realistas según la zona"""
        if zona == 'Pampeana':
            return np.random.exponential(70, 12)
        elif zona == 'NOA':
            return np.random.exponential(40, 12)
        elif zona == 'NEA':
            return np.random.exponential(120, 12)
        elif zona == 'Cuyo':
            return np.random.exponential(30, 12)
        else:  # Patagonia
            return np.random.exponential(50, 12)
    
    def _determinar_semilla_optima(self, zona, ph_suelo, temperatura, precipitacion, metros_cuadrados, precios):
        """
        Lógica mejorada para determinar semilla óptima con distribución balanceada
        """
        temp_media = np.mean(temperatura)
        precip_media = np.mean(precipitacion)
        precio_medio = np.mean(precios)
        
        # Lógica basada en condiciones argentinas con mejor distribución
        if zona == 'Pampeana':
            if 6.0 <= ph_suelo <= 7.5 and temp_media > 20 and precip_media > 600:
                return 'Soja' if precio_medio > 400 else 'Maíz'
            elif temp_media < 18 and precip_media > 400:
                return 'Trigo'
            else:
                return 'Girasol'
        elif zona == 'NOA':
            if temp_media > 25 and precip_media < 300:
                return 'Sorgo'
            elif temp_media > 22 and precip_media < 500:
                return 'Maíz'
            else:
                return 'Soja'
        elif zona == 'NEA':
            if precip_media > 800:
                return 'Soja'
            elif temp_media > 23:
                return 'Sorgo'
            else:
                return 'Maíz'
        elif zona == 'Cuyo':
            if temp_media < 18:
                return 'Cebada'
            elif precip_media < 400:
                return 'Girasol'
            else:
                return 'Trigo'
        else:  # Patagonia
            if temp_media < 12:
                return 'Cebada'
            elif temp_media < 16:
                return 'Trigo'
            else:
                return 'Girasol'
    
    def preparar_datos(self, df):
        """
        Prepara los datos para el modelo GRU
        """
        # Codificar zona
        df_encoded = df.copy()
        df_encoded['zona_encoded'] = self.label_encoder.fit_transform(df['zona'])
        
        # Preparar características estáticas
        static_features = ['metros_cuadrados', 'ph_suelo', 'materia_organica', 
                          'nitrogeno', 'fosforo', 'potasio', 'zona_encoded']
        
        # Preparar secuencias temporales
        temporal_features = []
        for mes in range(1, 13):
            temporal_features.extend([
                f'temp_mes_{mes}', f'precip_mes_{mes}', 
                f'humedad_mes_{mes}', f'precio_mes_{mes}'
            ])
        
        # Combinar características
        X_static = df_encoded[static_features].values
        X_temporal = df_encoded[temporal_features].values
        
        # Reshape temporal data para GRU (samples, timesteps, features)
        X_temporal = X_temporal.reshape(X_temporal.shape[0], 12, 4)
        
        # Normalizar datos
        X_static_scaled = self.scaler.fit_transform(X_static)
        X_temporal_scaled = np.zeros_like(X_temporal)
        
        # Normalizar cada característica temporal por separado
        for i in range(X_temporal.shape[2]):
            temporal_scaler = StandardScaler()
            X_temporal_scaled[:, :, i] = temporal_scaler.fit_transform(X_temporal[:, :, i])
        
        # Preparar variable objetivo
        y = self.label_encoder.fit_transform(df['semilla_optima'])
        
        return X_static_scaled, X_temporal_scaled, y
    
    def crear_modelo(self, n_static_features, n_temporal_features, n_classes):
        """
        Crea el modelo GRU híbrido
        """
        # Input para características estáticas
        static_input = tf.keras.Input(shape=(n_static_features,), name='static_input')
        static_dense = Dense(64, activation='relu')(static_input)
        static_dense = BatchNormalization()(static_dense)
        static_dense = Dropout(0.3)(static_dense)
        
        # Input para secuencias temporales
        temporal_input = tf.keras.Input(shape=(12, n_temporal_features), name='temporal_input')
        
        # Capas GRU
        gru1 = GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(temporal_input)
        gru1 = BatchNormalization()(gru1)
        
        gru2 = GRU(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(gru1)
        gru2 = BatchNormalization()(gru2)
        
        # Combinar características estáticas y temporales
        combined = tf.keras.layers.concatenate([static_dense, gru2])
        
        # Capas densas finales
        dense1 = Dense(128, activation='relu')(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        # Capa de salida
        output = Dense(n_classes, activation='softmax', name='output')(dense2)
        
        # Crear modelo
        model = tf.keras.Model(inputs=[static_input, temporal_input], outputs=output)
        
        return model
    
    def entrenar(self, X_static, X_temporal, y, epochs=100, batch_size=32):
        """
        Entrena el modelo GRU con manejo de clases desbalanceadas
        """
        # Verificar distribución de clases
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"Distribución de clases: {dict(zip(unique_classes, counts))}")
        
        # Filtrar clases con muy pocas muestras
        min_samples = 2  # Mínimo para stratify
        valid_indices = []
        
        for class_idx in unique_classes:
            class_indices = np.where(y == class_idx)[0]
            if len(class_indices) >= min_samples:
                valid_indices.extend(class_indices)
        
        if len(valid_indices) < len(y):
            print(f"Eliminando {len(y) - len(valid_indices)} muestras de clases con muy pocas observaciones")
            X_static = X_static[valid_indices]
            X_temporal = X_temporal[valid_indices]
            y = y[valid_indices]
        
        # Dividir datos
        X_static_train, X_static_test, X_temporal_train, X_temporal_test, y_train, y_test = train_test_split(
            X_static, X_temporal, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Crear modelo
        n_classes = len(np.unique(y))
        self.model = self.crear_modelo(X_static.shape[1], X_temporal.shape[2], n_classes)
        
        # Compilar modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Entrenar
        history = self.model.fit(
            [X_static_train, X_temporal_train], y_train,
            validation_data=([X_static_test, X_temporal_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history, X_static_test, X_temporal_test, y_test
    
    def evaluar(self, X_static_test, X_temporal_test, y_test):
        """
        Evalúa el modelo
        """
        # Predicciones
        y_pred = self.model.predict([X_static_test, X_temporal_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"Precisión del modelo: {accuracy:.4f}")
        
        # Reporte detallado
        semillas = self.label_encoder.classes_
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred_classes, target_names=semillas))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=semillas, yticklabels=semillas)
        plt.title('Matriz de Confusión - Recomendación de Semillas')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.show()
        
        return accuracy, y_pred_classes
    
    def predecir_semilla(self, zona, metros_cuadrados, ph_suelo, materia_organica,
                        nitrogeno, fosforo, potasio, datos_climaticos, precios):
        """
        Predice la mejor semilla para condiciones específicas
        """
        # Preparar datos estáticos
        zona_encoded = self.label_encoder.transform([zona])[0]
        static_data = np.array([[metros_cuadrados, ph_suelo, materia_organica,
                               nitrogeno, fosforo, potasio, zona_encoded]])
        static_data_scaled = self.scaler.transform(static_data)
        
        # Preparar datos temporales
        temporal_data = np.array(datos_climaticos + precios).reshape(1, 12, 4)
        
        # Normalizar (simplificado para el ejemplo)
        temporal_data_scaled = (temporal_data - np.mean(temporal_data, axis=1, keepdims=True)) / np.std(temporal_data, axis=1, keepdims=True)
        
        # Predicción
        prediccion = self.model.predict([static_data_scaled, temporal_data_scaled])
        semilla_idx = np.argmax(prediccion)
        confianza = np.max(prediccion)
        
        semilla_recomendada = self.label_encoder.inverse_transform([semilla_idx])[0]
        
        return semilla_recomendada, confianza

# Ejemplo de uso
def main():
    print("=== Sistema de Recomendación de Semillas - Agricultura Argentina ===\n")
    
    # Crear instancia del modelo
    ag_gru = AgriculturaGRU()
    
    # Generar datos simulados
    print("1. Generando datos simulados...")
    df = ag_gru.generar_datos_simulados(n_samples=3000)
    print(f"Datos generados: {len(df)} registros")
    print(f"Distribución de semillas:")
    print(df['semilla_optima'].value_counts())
    
    # Preparar datos
    print("\n2. Preparando datos para el modelo...")
    X_static, X_temporal, y = ag_gru.preparar_datos(df)
    print(f"Forma de datos estáticos: {X_static.shape}")
    print(f"Forma de datos temporales: {X_temporal.shape}")
    
    # Entrenar modelo
    print("\n3. Entrenando modelo GRU...")
    history, X_static_test, X_temporal_test, y_test = ag_gru.entrenar(
        X_static, X_temporal, y, epochs=50, batch_size=32
    )
    
    # Evaluar modelo
    print("\n4. Evaluando modelo...")
    accuracy, y_pred = ag_gru.evaluar(X_static_test, X_temporal_test, y_test)
    
    # Ejemplo de predicción
    print("\n5. Ejemplo de predicción:")
    # Datos de ejemplo para zona Pampeana
    zona_ejemplo = 'Pampeana'
    metros_cuadrados_ejemplo = 25000
    ph_suelo_ejemplo = 6.8
    materia_organica_ejemplo = 3.2
    nitrogeno_ejemplo = 35
    fosforo_ejemplo = 25
    potasio_ejemplo = 280
    
    # Datos climáticos ejemplo (12 meses: temp, precip, humedad)
    datos_climaticos_ejemplo = [
        [22, 85, 65], [24, 90, 70], [20, 75, 60], [18, 60, 55],
        [15, 40, 50], [12, 30, 45], [11, 25, 40], [13, 35, 45],
        [16, 50, 55], [19, 70, 60], [21, 80, 65], [23, 85, 70]
    ]
    
    # Precios ejemplo (12 meses)
    precios_ejemplo = [450, 460, 440, 435, 455, 465, 470, 455, 445, 450, 460, 465]
    
    # Flatten datos climáticos
    datos_climaticos_flat = [item for sublist in datos_climaticos_ejemplo for item in sublist]
    
    semilla_recomendada, confianza = ag_gru.predecir_semilla(
        zona_ejemplo, metros_cuadrados_ejemplo, ph_suelo_ejemplo, 
        materia_organica_ejemplo, nitrogeno_ejemplo, fosforo_ejemplo, 
        potasio_ejemplo, datos_climaticos_flat, precios_ejemplo
    )
    
    print(f"Zona: {zona_ejemplo}")
    print(f"Área: {metros_cuadrados_ejemplo} m²")
    print(f"pH del suelo: {ph_suelo_ejemplo}")
    print(f"Semilla recomendada: {semilla_recomendada}")
    print(f"Confianza: {confianza:.4f}")
    
    # Mostrar arquitectura del modelo
    print("\n6. Arquitectura del modelo:")
    ag_gru.model.summary()

if __name__ == "__main__":
    main()