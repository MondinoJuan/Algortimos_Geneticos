import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ast
import warnings
warnings.filterwarnings('ignore')

class LSTMCropPredictor:
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.sequence_length = 1  # Para datos agrícolas, usaremos secuencias simples
        
    def parse_array_columns(self, df):
        """Procesa las columnas que contienen arrays como strings"""
        array_columns = ['temperatura_media_C', 'humedad_relativa_%', 
                        'velocidad_viento_m_s', 'velocidad_viento_km_h', 
                        'precipitacion_mm_mes']
        
        df_processed = df.copy()
        
        def safe_parse_array(x):
            """Parsea arrays de forma segura"""
            if pd.isna(x) or x in ['SD', 'sd', 'N/A', 'NA', '', ' ']:
                return None
            try:
                parsed = ast.literal_eval(str(x))
                # Verificar que es una lista y contiene números
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Limpiar valores no numéricos dentro del array
                    numeric_values = []
                    for val in parsed:
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            numeric_values.append(float(val))
                    return numeric_values if numeric_values else None
                return None
            except:
                return None
        
        for col in array_columns:
            if col in df_processed.columns:
                # Convertir string array a lista y extraer estadísticas
                parsed_arrays = df_processed[col].apply(safe_parse_array)
                
                df_processed[col + '_mean'] = parsed_arrays.apply(lambda x: np.mean(x) if x and len(x) > 0 else np.nan)
                df_processed[col + '_std'] = parsed_arrays.apply(lambda x: np.std(x) if x and len(x) > 1 else 0.0)
                df_processed[col + '_min'] = parsed_arrays.apply(lambda x: np.min(x) if x and len(x) > 0 else np.nan)
                df_processed[col + '_max'] = parsed_arrays.apply(lambda x: np.max(x) if x and len(x) > 0 else np.nan)
        
        # Eliminar columnas originales de arrays
        columns_to_drop = [col for col in array_columns if col in df_processed.columns]
        df_processed = df_processed.drop(columns=columns_to_drop)
        
        return df_processed
    
    def encode_categorical_features(self, df, fit=True):
        """Codifica variables categóricas"""
        df_encoded = df.copy()
        categorical_cols = ['cultivo_nombre', 'departamento_nombre']
        
        for col in categorical_cols:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Para nuevos datos, usar transform
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                
        return df_encoded
    
    def clean_data(self, df):
        """Limpia los datos manejando valores faltantes y no válidos"""
        print("Limpiando datos...")
        df_clean = df.copy()
        
        # Reemplazar valores 'SD' (Sin Datos) y otros valores no numéricos con NaN
        non_numeric_indicators = ['SD', 'sd', 'N/A', 'NA', 'null', 'NULL', '', ' ']
        
        # Identificar columnas numéricas (excluyendo las categóricas)
        categorical_cols = ['cultivo_nombre', 'departamento_nombre']
        numeric_cols = [col for col in df_clean.columns if col not in categorical_cols]
        
        # Limpiar columnas numéricas
        for col in numeric_cols:
            if col != 'produccion_tn':  # No limpiar el target todavía
                # Reemplazar indicadores de datos faltantes con NaN
                df_clean[col] = df_clean[col].replace(non_numeric_indicators, np.nan)
                
                # Si es una columna de array, manejar especialmente
                array_columns = ['temperatura_media_C', 'humedad_relativa_%', 
                               'velocidad_viento_m_s', 'velocidad_viento_km_h', 
                               'precipitacion_mm_mes']
                if col in array_columns:
                    continue  # Se maneja en parse_array_columns
                
                # Convertir a numérico, forzando errores a NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Limpiar target (produccion_tn) - más estricto
        df_clean['produccion_tn'] = df_clean['produccion_tn'].replace(non_numeric_indicators, np.nan)
        df_clean['produccion_tn'] = pd.to_numeric(df_clean['produccion_tn'], errors='coerce')
        
        # Eliminar filas donde el target es NaN, 0, negativo o infinito
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['produccion_tn'])
        df_clean = df_clean[df_clean['produccion_tn'] > 0]  # Solo valores positivos
        df_clean = df_clean[np.isfinite(df_clean['produccion_tn'])]  # Solo valores finitos
        rows_removed_target = initial_rows - len(df_clean)
        
        if rows_removed_target > 0:
            print(f"Se eliminaron {rows_removed_target} filas por target faltante")
        
        # Eliminar filas con demasiados valores faltantes (más del 50% de las columnas)
        threshold = len(df_clean.columns) * 0.5
        df_clean = df_clean.dropna(thresh=threshold)
        rows_after_threshold = len(df_clean)
        
        if initial_rows - rows_removed_target - rows_after_threshold > 0:
            print(f"Se eliminaron {initial_rows - rows_removed_target - rows_after_threshold} filas adicionales por exceso de valores faltantes")
        
        print(f"Datos después de limpieza: {df_clean.shape}")
        
        return df_clean

    def prepare_data(self, csv_path):
        """Carga y prepara los datos"""
        print("Cargando datos...")
        df = pd.read_csv(csv_path)
        
        print(f"Forma original de los datos: {df.shape}")
        print(f"Cultivos únicos: {df['cultivo_nombre'].nunique()}")
        print(f"Departamentos únicos: {df['departamento_nombre'].nunique()}")
        
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Procesar columnas de arrays
        df_processed = self.parse_array_columns(df_clean)
        
        # Manejar valores faltantes después del procesamiento
        # Rellenar con la mediana para columnas numéricas
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        # Codificar variables categóricas
        df_encoded = self.encode_categorical_features(df_processed, fit=True)
        
        # Verificar que no hay más NaN
        if df_encoded.isnull().sum().sum() > 0:
            print("Advertencia: Aún hay valores NaN después del preprocesamiento")
            print(df_encoded.isnull().sum())
        
        # Separar features y target
        target_col = 'produccion_tn'
        feature_cols = [col for col in df_encoded.columns if col != target_col]
        
        X = df_encoded[feature_cols].values
        y = df_encoded[target_col].values.reshape(-1, 1)
        
        # Verificar que no hay valores infinitos o NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalizar datos
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_target.fit_transform(y)
        
        return X_scaled, y_scaled.flatten(), df_encoded
    
    def create_sequences(self, X, y):
        """Crea secuencias para LSTM"""
        # Para datos agrícolas, cada muestra es independiente
        # Reformateamos para LSTM: (samples, timesteps, features)
        X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
        return X_seq, y
    
    def build_model(self, input_shape):
        """Construye el modelo LSTM"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Entrena el modelo LSTM"""
        print("Construyendo modelo LSTM...")
        
        # Crear secuencias
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        # Construir modelo
        self.model = self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        print("Arquitectura del modelo:")
        self.model.summary()
        
        # Callbacks con early stopping de 15 épocas
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]
        
        print("Entrenando modelo...")
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Realiza predicciones"""
        X_seq, _ = self.create_sequences(X, np.zeros(X.shape[0]))
        predictions_scaled = self.model.predict(X_seq)
        predictions = self.scaler_target.inverse_transform(predictions_scaled.reshape(-1, 1))
        return predictions.flatten()
    
    def evaluate_model(self, X_test, y_test):
        """Evalúa el modelo"""
        predictions = self.predict(X_test)
        y_test_original = self.scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)
        
        print(f"\nMétricas de evaluación:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actual': y_test_original
        }
    
    def plot_training_history(self, history):
        """Grafica la historia del entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Entrenamiento')
        ax1.plot(history.history['val_loss'], label='Validación')
        ax1.set_title('Pérdida durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(history.history['mae'], label='Entrenamiento')
        ax2.plot(history.history['val_mae'], label='Validación')
        ax2.set_title('Error Absoluto Medio durante el entrenamiento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, results, sample_size=500):
        """Grafica comparación entre valores reales y predicciones"""
        predictions = results['predictions']
        actual = results['actual']
        
        # Verificar que no hay valores infinitos o NaN
        finite_mask = np.isfinite(predictions) & np.isfinite(actual) & (actual > 0)
        predictions_clean = predictions[finite_mask]
        actual_clean = actual[finite_mask]
        
        if len(predictions_clean) == 0:
            print("Error: No hay datos válidos para graficar después de la limpieza.")
            return
        
        print(f"Graficando {len(predictions_clean)} puntos válidos de {len(predictions)} totales")
        
        # Tomar una muestra si hay muchos datos
        if len(predictions_clean) > sample_size:
            indices = np.random.choice(len(predictions_clean), sample_size, replace=False)
            predictions_sample = predictions_clean[indices]
            actual_sample = actual_clean[indices]
        else:
            predictions_sample = predictions_clean
            actual_sample = actual_clean
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico de dispersión
        ax1.scatter(actual_sample, predictions_sample, alpha=0.6, color='blue', s=20)
        ax1.plot([actual_sample.min(), actual_sample.max()], 
                [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
        ax1.set_xlabel('Valores Reales (toneladas)')
        ax1.set_ylabel('Predicciones (toneladas)')
        ax1.set_title('Predicciones vs Valores Reales')
        ax1.grid(True)
        
        # Error relativo (mejorado para evitar infinitos)
        # Agregar un pequeño epsilon para evitar división por cero
        epsilon = 1e-8
        actual_safe = np.maximum(actual_sample, epsilon)
        relative_error = np.abs((predictions_sample - actual_sample) / actual_safe) * 100
        
        # Limitar errores relativos extremos para mejor visualización
        relative_error = np.clip(relative_error, 0, 500)  # Max 500% de error
        
        ax2.hist(relative_error, bins=50, alpha=0.7, color='green', range=(0, np.percentile(relative_error, 95)))
        ax2.set_xlabel('Error Relativo (%)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución del Error Relativo')
        ax2.grid(True)
        
        # Series de tiempo (primeras 100 muestras)
        sample_100 = min(100, len(predictions_sample))
        x_axis = range(sample_100)
        ax3.plot(x_axis, actual_sample[:sample_100], 'bo-', label='Real', markersize=4, linewidth=1)
        ax3.plot(x_axis, predictions_sample[:sample_100], 'ro-', label='Predicción', markersize=4, linewidth=1)
        ax3.set_xlabel('Muestra')
        ax3.set_ylabel('Producción (toneladas)')
        ax3.set_title('Comparación Serie de Tiempo (primeras 100 muestras)')
        ax3.legend()
        ax3.grid(True)
        
        # Residuos
        residuals = predictions_sample - actual_sample
        ax4.scatter(predictions_sample, residuals, alpha=0.6, color='purple', s=20)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicciones (toneladas)')
        ax4.set_ylabel('Residuos (toneladas)')
        ax4.set_title('Gráfico de Residuos')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas adicionales (con datos limpios)
        finite_relative_error = relative_error[np.isfinite(relative_error)]
        print(f"\nEstadísticas adicionales:")
        print(f"Puntos válidos para análisis: {len(predictions_clean)}/{len(predictions)} ({len(predictions_clean)/len(predictions)*100:.1f}%)")
        print(f"Error relativo promedio: {np.mean(finite_relative_error):.2f}%")
        print(f"Error relativo mediana: {np.median(finite_relative_error):.2f}%")
        print(f"Error relativo percentil 90: {np.percentile(finite_relative_error, 90):.2f}%")
        print(f"Rango de valores reales: [{actual_clean.min():.2f}, {actual_clean.max():.2f}]")
        print(f"Rango de predicciones: [{predictions_clean.min():.2f}, {predictions_clean.max():.2f}]")
        
        # Detectar y reportar problemas
        zero_actual = np.sum(actual == 0)
        negative_actual = np.sum(actual < 0)
        inf_predictions = np.sum(~np.isfinite(predictions))
        
        if zero_actual > 0:
            print(f"Advertencia: {zero_actual} valores reales son exactamente 0")
        if negative_actual > 0:
            print(f"Advertencia: {negative_actual} valores reales son negativos")
        if inf_predictions > 0:
            print(f"Advertencia: {inf_predictions} predicciones son infinitas o NaN")

def main():
    """Función principal para ejecutar el pipeline completo"""
    # Inicializar predictor
    predictor = LSTMCropPredictor()
    
    # Solicitar ruta del archivo
    csv_path = "df_con_prod.csv"
    
    try:
        # Preparar datos
        X, y, df_processed = predictor.prepare_data(csv_path)
        
        # Verificar que tenemos datos suficientes
        if len(X) < 100:
            print(f"Advertencia: Solo {len(X)} muestras disponibles después de la limpieza.")
            print("Se recomienda tener al menos 1000 muestras para un buen entrenamiento.")
        
        # Dividir datos en entrenamiento (70%), validación (15%) y prueba (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"\nDivisión de datos:")
        print(f"Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validación: {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
        
        # Entrenar modelo
        history = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # Mostrar historia del entrenamiento
        predictor.plot_training_history(history)
        
        # Evaluar modelo
        results = predictor.evaluate_model(X_test, y_test)
        
        # Mostrar gráficas comparativas
        predictor.plot_predictions(results)
        
        # Análisis por cultivo
        print("\nAnálisis por cultivo:")
        df_test = df_processed.iloc[X_test.shape[0]*-1:]  # Aproximación para datos de test
        
        return predictor, results
        
    except FileNotFoundError:
        print("Archivo no encontrado. Verifica la ruta.")
        return None, None
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Ejecutar si es el script principal
if __name__ == "__main__":
    result = main()
    
    if result[0] is not None:
        predictor, results = result
        print("\n" + "="*60)
        print("RESUMEN DEL MODELO LSTM PARA PREDICCIÓN DE PRODUCCIÓN AGRÍCOLA")
        print("="*60)
        print("El modelo ha sido entrenado exitosamente.")
        print("Puedes usar 'predictor' para hacer nuevas predicciones.")
        print("Los resultados están disponibles en 'results'.")
    else:
        print("\nEl entrenamiento no se pudo completar debido a errores en los datos.")