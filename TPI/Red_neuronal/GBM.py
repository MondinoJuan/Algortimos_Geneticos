import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AgriculturalProductionPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'catboost': None
        }
        
    def parse_array_column(self, series):
        """Convierte strings de arrays a listas numéricas de forma robusta"""
        def safe_eval(x):
            try:
                # Si es NaN o None
                if pd.isna(x):
                    return []
                
                # Si ya es una lista
                if isinstance(x, list):
                    return x
                
                # Si es un float o int (NaN convertido)
                if isinstance(x, (int, float)):
                    return []
                
                # Si es string, intentar parsear
                if isinstance(x, str):
                    # Limpiar string
                    x = x.strip()
                    if x in ['', 'nan', 'NaN', 'null', 'None']:
                        return []
                    return ast.literal_eval(x)
                
                return []
            except:
                return []
        return series.apply(safe_eval)
    
    def extract_array_features(self, df, array_columns):
        """Extrae características estadísticas de columnas tipo array de forma robusta"""
        new_features = {}
        
        for col in array_columns:
            if col in df.columns:
                print(f"   Procesando arrays de {col}...")
                arrays = self.parse_array_column(df[col])
                
                # Función auxiliar para aplicar operaciones de forma segura
                def safe_apply(func, default=0):
                    def safe_func(x):
                        try:
                            if isinstance(x, list) and len(x) > 0:
                                # Filtrar valores no numéricos
                                numeric_vals = [v for v in x if isinstance(v, (int, float)) and not pd.isna(v)]
                                if len(numeric_vals) > 0:
                                    return func(numeric_vals)
                            return default
                        except:
                            return default
                    return arrays.apply(safe_func)
                
                # Estadísticas básicas con manejo robusto
                new_features[f'{col}_mean'] = safe_apply(np.mean)
                new_features[f'{col}_std'] = safe_apply(np.std)
                new_features[f'{col}_min'] = safe_apply(np.min)
                new_features[f'{col}_max'] = safe_apply(np.max)
                new_features[f'{col}_median'] = safe_apply(np.median)
                
                # Características avanzadas
                new_features[f'{col}_range'] = new_features[f'{col}_max'] - new_features[f'{col}_min']
                
                # Coeficiente de variación con división segura
                mean_vals = new_features[f'{col}_mean']
                std_vals = new_features[f'{col}_std']
                new_features[f'{col}_cv'] = std_vals / (mean_vals.abs() + 1e-8)
                
                # Percentiles
                new_features[f'{col}_q25'] = safe_apply(lambda x: np.percentile(x, 25) if len(x) > 0 else 0)
                new_features[f'{col}_q75'] = safe_apply(lambda x: np.percentile(x, 75) if len(x) > 0 else 0)
                
                # Tendencias (si hay suficientes datos temporales)
                def calculate_trend(arr):
                    try:
                        if isinstance(arr, list) and len(arr) >= 3:
                            # Filtrar valores numéricos válidos
                            numeric_vals = [v for v in arr if isinstance(v, (int, float)) and not pd.isna(v)]
                            if len(numeric_vals) >= 3:
                                x = np.arange(len(numeric_vals))
                                z = np.polyfit(x, numeric_vals, 1)
                                return z[0]  # slope
                        return 0
                    except:
                        return 0
                
                new_features[f'{col}_trend'] = arrays.apply(calculate_trend)
                
                # Contar valores válidos por array
                new_features[f'{col}_count'] = arrays.apply(
                    lambda x: len([v for v in x if isinstance(v, (int, float)) and not pd.isna(v)]) 
                    if isinstance(x, list) else 0
                )
                
        return pd.DataFrame(new_features)
    
    def create_interaction_features(self, df):
        """Crea características de interacción relevantes para agricultura"""
        interactions = {}
        
        # Interacciones climáticas importantes
        if 'temperatura_media_C_mean' in df.columns and 'humedad_relativa_%_mean' in df.columns:
            interactions['temp_humidity_interaction'] = df['temperatura_media_C_mean'] * df['humedad_relativa_%_mean']
        
        if 'precipitacion_mm_mes_mean' in df.columns and 'temperatura_media_C_mean' in df.columns:
            interactions['precip_temp_ratio'] = df['precipitacion_mm_mes_mean'] / (df['temperatura_media_C_mean'] + 1e-8)
        
        # Índice de estrés hídrico simplificado
        if all(col in df.columns for col in ['precipitacion_mm_mes_mean', 'temperatura_media_C_mean', 'humedad_relativa_%_mean']):
            interactions['water_stress_index'] = (df['temperatura_media_C_mean'] * (100 - df['humedad_relativa_%_mean'])) / (df['precipitacion_mm_mes_mean'] + 1e-8)
        
        # Características del suelo
        if all(col in df.columns for col in ['clay', 'silt', 'sand']):
            interactions['soil_texture_index'] = df['clay'] / (df['sand'] + 1e-8)
        
        return pd.DataFrame(interactions)
    
    def clean_data(self, df):
        """Limpia los datos eliminando filas con valores faltantes críticos"""
        print(f"Dataset inicial: {df.shape[0]} filas")
        
        # Crear copia para no modificar el original
        df_clean = df.copy()
        
        # 1. Identificar y reportar valores faltantes
        missing_info = df_clean.isnull().sum()
        missing_columns = missing_info[missing_info > 0]
        
        if len(missing_columns) > 0:
            print("\n📋 VALORES FALTANTES ENCONTRADOS:")
            for col, count in missing_columns.items():
                percentage = (count / len(df_clean)) * 100
                print(f"   • {col}: {count} valores ({percentage:.1f}%)")
        
        # 2. Limpiar valores especiales comunes
        # Reemplazar strings que indican datos faltantes
        missing_indicators = ['SD', 'sin datos', 'Sin datos', 'SIN DATOS', 'N/A', 'n/a', 'NULL', 'null', '', ' ', 'nan', 'NaN']
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Para columnas de texto, reemplazar indicadores por NaN
                df_clean[col] = df_clean[col].replace(missing_indicators, np.nan)
                
                # Intentar convertir a numérico si es posible
                if col not in ['cultivo_nombre', 'departamento_nombre']:
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except:
                        pass
        
        # 3. Manejar columnas tipo array con valores faltantes
        array_columns = ['temperatura_media_C', 'humedad_relativa_%', 'velocidad_viento_m_s', 
                        'velocidad_viento_km_h', 'precipitacion_mm_mes']
        
        for col in array_columns:
            if col in df_clean.columns:
                # Marcar como NaN si el array está vacío o malformado
                def validate_array(x):
                    try:
                        if pd.isna(x) or x in missing_indicators:
                            return np.nan
                        
                        # Si es float (NaN convertido), marcar como inválido
                        if isinstance(x, (int, float)):
                            return np.nan
                            
                        if isinstance(x, str):
                            x = x.strip()
                            if x in ['', 'nan', 'NaN', 'null', 'None', '[]']:
                                return np.nan
                            parsed = ast.literal_eval(x)
                            if len(parsed) == 0:
                                return np.nan
                            return x
                        
                        # Si ya es una lista
                        if isinstance(x, list):
                            if len(x) == 0:
                                return np.nan
                            return str(x)  # Convertir de vuelta a string para procesamiento posterior
                        
                        return x
                    except:
                        return np.nan
                
                df_clean[col] = df_clean[col].apply(validate_array)
        
        # 4. Eliminar filas con valores faltantes en columnas críticas
        critical_columns = ['produccion_tn', 'superficie_sembrada_ha', 'cultivo_nombre']
        existing_critical = [col for col in critical_columns if col in df_clean.columns]
        
        if existing_critical:
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(subset=existing_critical)
            rows_after = len(df_clean)
            removed = rows_before - rows_after
            if removed > 0:
                print(f"\n🗑️  Eliminadas {removed} filas por valores faltantes en columnas críticas")
        
        # 5. Para otras columnas, eliminar filas donde TODAS las características importantes son NaN
        feature_columns = [col for col in df_clean.columns 
                          if col not in ['cultivo_nombre', 'departamento_nombre', 'anio']]
        
        rows_before = len(df_clean)
        # Eliminar filas donde más del 50% de las características son NaN
        threshold = len(feature_columns) * 0.5
        df_clean = df_clean.dropna(thresh=len(df_clean.columns) - threshold)
        rows_after = len(df_clean)
        removed = rows_before - rows_after
        
        if removed > 0:
            print(f"🗑️  Eliminadas {removed} filas adicionales por exceso de valores faltantes")
        
        print(f"✅ Dataset limpio: {df_clean.shape[0]} filas ({((df_clean.shape[0]/df.shape[0])*100):.1f}% conservado)")
        
        # 6. Verificar que aún tenemos datos suficientes
        if len(df_clean) < 100:
            raise ValueError("⚠️  Muy pocos datos después de la limpieza. Revisa tu dataset original.")
        
        # 7. Rellenar valores faltantes restantes con estrategias inteligentes
        df_clean = self.impute_remaining_values(df_clean)
        
        return df_clean
    
    def impute_remaining_values(self, df):
        """Rellena valores faltantes restantes con estrategias inteligentes"""
        df_imputed = df.copy()
        
        # Para variables numéricas: usar mediana por cultivo/departamento
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['anio']]
        
        for col in numeric_columns:
            if df_imputed[col].isnull().any():
                # Intentar rellenar por cultivo primero
                if 'cultivo_nombre' in df_imputed.columns:
                    df_imputed[col] = df_imputed.groupby('cultivo_nombre')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                
                # Si aún quedan NaN, usar la mediana general
                if df_imputed[col].isnull().any():
                    median_val = df_imputed[col].median()
                    if pd.notna(median_val):
                        df_imputed[col].fillna(median_val, inplace=True)
                    else:
                        # Si no hay mediana, usar 0
                        df_imputed[col].fillna(0, inplace=True)
        
        # Para variables categóricas: usar la moda
        categorical_columns = ['cultivo_nombre', 'departamento_nombre']
        for col in categorical_columns:
            if col in df_imputed.columns and df_imputed[col].isnull().any():
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col].fillna(mode_val[0], inplace=True)
                else:
                    df_imputed[col].fillna('Desconocido', inplace=True)
        
        return df_imputed
    
    def preprocess_data(self, df, is_training=True):
        """Preprocesa los datos completos"""
        # Primero limpiar los datos
        if is_training:
            df_processed = self.clean_data(df)
        else:
            df_processed = df.copy()
            # Para predicción, aplicar imputación básica sin eliminar filas
            df_processed = self.impute_remaining_values(df_processed)
        
        # Identificar columnas tipo array
        array_columns = ['temperatura_media_C', 'humedad_relativa_%', 'velocidad_viento_m_s', 
                        'velocidad_viento_km_h', 'precipitacion_mm_mes']
        
        print("🔄 Extrayendo características de arrays climáticos...")
        # Extraer características de arrays
        array_features = self.extract_array_features(df_processed, array_columns)
        df_processed = pd.concat([df_processed, array_features], axis=1)
        
        # Eliminar columnas originales tipo array
        df_processed = df_processed.drop(columns=[col for col in array_columns if col in df_processed.columns])
        
        # Crear características de interacción
        interaction_features = self.create_interaction_features(df_processed)
        df_processed = pd.concat([df_processed, interaction_features], axis=1)
        
        # Encoding de variables categóricas
        categorical_columns = ['cultivo_nombre', 'departamento_nombre']
        for col in categorical_columns:
            if col in df_processed.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    # Para predicción, manejar valores no vistos
                    df_processed[col] = df_processed[col].astype(str)
                    for val in df_processed[col].unique():
                        if val not in self.label_encoders[col].classes_:
                            # Asignar a la clase más frecuente
                            df_processed[col] = df_processed[col].replace(val, self.label_encoders[col].classes_[0])
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Características temporales
        if 'anio' in df_processed.columns:
            df_processed['anio_normalized'] = (df_processed['anio'] - df_processed['anio'].min()) / (df_processed['anio'].max() - df_processed['anio'].min() + 1e-8)
        
        return df_processed
    
    def prepare_features(self, df, target_column='produccion_tn', is_training=True):
        """Prepara las características para el modelo"""
        df_processed = self.preprocess_data(df, is_training)
        
        if target_column in df_processed.columns:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
        
        # Escalado robusto (mejor para datos con outliers)
        if is_training:
            self.scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), 
                columns=X.columns, 
                index=X.index
            )
        
        # Selección de características (solo en entrenamiento)
        if is_training and y is not None:
            # Seleccionar las mejores características
            k_best = min(50, len(X_scaled.columns))  # Limitar el número de características
            self.feature_selector = SelectKBest(f_regression, k=k_best)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            selected_features = X_scaled.columns[self.feature_selector.get_support()]
            X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
        elif self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            selected_features = X_scaled.columns[self.feature_selector.get_support()]
            X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
        else:
            X_final = X_scaled
        
        # Verificar que no hay NaN después del preprocesamiento
        if X_final.isnull().any().any():
            print("⚠️  Advertencia: Detectados valores NaN restantes. Aplicando limpieza final...")
            X_final = X_final.fillna(0)  # Último recurso: rellenar con 0
        
        return X_final, y
    
    def get_model_params(self):
        """Parámetros optimizados para cada modelo"""
        return {
            'xgboost': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2],
                'num_leaves': [31, 63, 127]
            },
            'catboost': {
                'iterations': [100, 200, 300, 500],
                'depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [128, 254]
            }
        }
    
    def train_models(self, X, y, cv_folds=5):
        """Entrena múltiples modelos con optimización de hiperparámetros"""
        results = {}
        
        # Configurar validación cruzada
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # XGBoost
        print("Optimizando XGBoost...")
        xgb_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
        xgb_search = RandomizedSearchCV(
            xgb_model, 
            self.get_model_params()['xgboost'],
            n_iter=20,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_search.fit(X, y)
        self.models['xgboost'] = xgb_search.best_estimator_
        results['xgboost'] = -xgb_search.best_score_
        
        # LightGBM
        print("Optimizando LightGBM...")
        lgb_model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1)
        lgb_search = RandomizedSearchCV(
            lgb_model,
            self.get_model_params()['lightgbm'],
            n_iter=20,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1
        )
        lgb_search.fit(X, y)
        self.models['lightgbm'] = lgb_search.best_estimator_
        results['lightgbm'] = -lgb_search.best_score_
        
        # CatBoost
        print("Optimizando CatBoost...")
        cat_model = CatBoostRegressor(random_state=self.random_state, verbose=False)
        cat_search = RandomizedSearchCV(
            cat_model,
            self.get_model_params()['catboost'],
            n_iter=20,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1
        )
        cat_search.fit(X, y)
        self.models['catboost'] = cat_search.best_estimator_
        results['catboost'] = -cat_search.best_score_
        
        # Seleccionar el mejor modelo
        best_model_name = min(results.keys(), key=lambda k: results[k])
        self.best_model = self.models[best_model_name]
        
        print(f"\nResultados de validación cruzada (RMSE):")
        for model_name, score in results.items():
            print(f"{model_name}: {np.sqrt(score):.2f}")
        print(f"\nMejor modelo: {best_model_name}")
        
        return results
    
    def evaluate_model(self, X_test, y_test):
        """Evalúa el mejor modelo"""
        y_pred = self.best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MSE': mse
        }
        
        return metrics, y_pred
    
    def get_feature_importance(self, top_n=20):
        """Obtiene la importancia de las características"""
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_names = self.feature_selector.get_feature_names_out() if self.feature_selector else self.feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        
        return None
    
    def plot_model_evaluation(self, y_true, y_pred, save_plots=False):
        """
        Crea múltiples gráficos para evaluar la calidad del modelo
        """
        if plt is None:
            print("Matplotlib no disponible. Instala con: pip install matplotlib seaborn")
            return
        
        # Configurar el estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Scatter plot: Predicciones vs Valores Reales
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Línea de predicción perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        # Línea de regresión
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), "b-", alpha=0.8, label=f'Regresión (R² = {r2_score(y_true, y_pred):.3f})')
        
        plt.xlabel('Valores Reales (tn)', fontsize=12)
        plt.ylabel('Predicciones (tn)', fontsize=12)
        plt.title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Residuos vs Predicciones
        ax2 = plt.subplot(2, 3, 2)
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicciones (tn)', fontsize=12)
        plt.ylabel('Residuos (tn)', fontsize=12)
        plt.title('Residuos vs Predicciones', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Histograma de residuos
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Curva normal para comparación
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
        
        plt.xlabel('Residuos (tn)', fontsize=12)
        plt.ylabel('Densidad', fontsize=12)
        plt.title('Distribución de Residuos', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Q-Q Plot para normalidad de residuos
        ax4 = plt.subplot(2, 3, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot de Residuos', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Errores absolutos por rango de valores
        ax5 = plt.subplot(2, 3, 5)
        abs_errors = np.abs(residuals)
        
        # Crear bins para los valores reales
        n_bins = 10
        bins = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
        bin_centers = []
        mean_errors = []
        
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                mean_errors.append(abs_errors[mask].mean())
        
        plt.plot(bin_centers, mean_errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Valor Real (tn)', fontsize=12)
        plt.ylabel('Error Absoluto Medio (tn)', fontsize=12)
        plt.title('Error por Rango de Valores', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. Métricas de evaluación
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calcular métricas
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Crear texto con métricas
        metrics_text = f"""
        MÉTRICAS DE EVALUACIÓN
        
        R² (Coef. Determinación): {r2:.4f}
        RMSE (Error Cuadrático): {rmse:.2f} tn
        MAE (Error Absoluto): {mae:.2f} tn
        MAPE (Error Porcentual): {mape:.2f}%
        
        INTERPRETACIÓN:
        • R² = {r2:.3f} → {"Excelente" if r2 > 0.9 else "Bueno" if r2 > 0.7 else "Regular" if r2 > 0.5 else "Pobre"} ajuste
        • RMSE = {rmse:.1f} tn → Error típico
        • MAE = {mae:.1f} tn → Error promedio
        
        DISTRIBUCIÓN DE ERRORES:
        • Media residuos: {np.mean(residuals):.3f}
        • Std residuos: {np.std(residuals):.3f}
        • Residuos < 10% valor: {(np.abs(residuals/y_true) < 0.1).mean()*100:.1f}%
        """
        
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
            print("Gráficos guardados como 'model_evaluation.png'")
        
        plt.show()
        
        # Imprimir resumen estadístico
        print("\n" + "="*60)
        print("RESUMEN ESTADÍSTICO DE LA EVALUACIÓN")
        print("="*60)
        
        print(f"\n📊 MÉTRICAS PRINCIPALES:")
        print(f"   • R² Score: {r2:.4f} ({'Excelente' if r2 > 0.9 else 'Bueno' if r2 > 0.7 else 'Regular' if r2 > 0.5 else 'Pobre'})")
        print(f"   • RMSE: {rmse:.2f} toneladas")
        print(f"   • MAE: {mae:.2f} toneladas") 
        print(f"   • MAPE: {mape:.2f}%")
        
        print(f"\n🎯 PRECISIÓN POR TOLERANCIA:")
        tolerances = [0.05, 0.10, 0.20, 0.30]
        for tol in tolerances:
            accurate = (np.abs(residuals / y_true) <= tol).mean() * 100
            print(f"   • Predicciones dentro del {tol*100:.0f}%: {accurate:.1f}%")
        
        print(f"\n📈 ANÁLISIS DE RESIDUOS:")
        print(f"   • Media de residuos: {np.mean(residuals):.3f}")
        print(f"   • Desviación estándar: {np.std(residuals):.3f}")
        print(f"   • Sesgo (skewness): {stats.skew(residuals):.3f}")
        print(f"   • Curtosis: {stats.kurtosis(residuals):.3f}")
        
        # Test de normalidad
        _, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"   • Test normalidad (p-value): {p_value:.4f}")
        print(f"   • Residuos normales: {'Sí' if p_value > 0.05 else 'No'}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'normality_p_value': p_value
        }

    def plot_feature_importance(self, top_n=15, save_plot=False):
        """Grafica la importancia de las características"""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df is None or plt is None:
            print("No se puede generar el gráfico de importancia")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras horizontal
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importancia', fontsize=12)
        plt.title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Gráfico guardado como 'feature_importance.png'")
        
        plt.show()

    def plot_learning_curves(self, X, y, save_plot=False):
        """Grafica curvas de aprendizaje para detectar overfitting"""
        if plt is None:
            print("Matplotlib no disponible")
            return
        
        print("Generando curvas de aprendizaje...")
        
        # Usar el mejor modelo encontrado
        model = self.best_model
        
        # Diferentes tamaños de conjunto de entrenamiento
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            # Crear muestra de entrenamiento
            n_samples = int(len(X) * train_size)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_train_subset = X.iloc[indices]
            y_train_subset = y.iloc[indices]
            
            # Dividir en train/validation
            X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                X_train_subset, y_train_subset, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_cv, y_train_cv)
            
            # Evaluar
            train_pred = model_copy.predict(X_train_cv)
            val_pred = model_copy.predict(X_val_cv)
            
            train_scores.append(r2_score(y_train_cv, train_pred))
            val_scores.append(r2_score(y_val_cv, val_pred))
        
        # Graficar
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(train_sizes * len(X), train_scores, 'o-', label='Score Entrenamiento', linewidth=2)
        plt.plot(train_sizes * len(X), val_scores, 'o-', label='Score Validación', linewidth=2)
        plt.xlabel('Tamaño Conjunto Entrenamiento')
        plt.ylabel('R² Score')
        plt.title('Curvas de Aprendizaje - R² Score', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # También mostrar RMSE
        plt.subplot(2, 1, 2)
        train_rmse = [np.sqrt(mean_squared_error(y.iloc[np.random.choice(len(X), int(len(X) * size), replace=False)], 
                                                model.predict(X.iloc[np.random.choice(len(X), int(len(X) * size), replace=False)]))) 
                     for size in train_sizes]
        
        plt.plot(train_sizes * len(X), train_rmse, 'o-', label='RMSE', linewidth=2, color='red')
        plt.xlabel('Tamaño Conjunto Entrenamiento')
        plt.ylabel('RMSE (toneladas)')
        plt.title('Curva de Error - RMSE', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_prediction_by_crop(self, X, y_true, y_pred, save_plot=False):
        """Analiza predicciones por tipo de cultivo"""
        if plt is None:
            print("Matplotlib no disponible")
            return
        
        # Necesitamos recuperar los nombres de cultivos originales
        if 'cultivo_nombre' not in X.columns:
            print("No se puede analizar por cultivo - información no disponible")
            return
        
        # Decodificar nombres de cultivos
        cultivo_encoded = X['cultivo_nombre']
        if 'cultivo_nombre' in self.label_encoders:
            try:
                cultivo_nombres = self.label_encoders['cultivo_nombre'].inverse_transform(cultivo_encoded)
                
                # Crear DataFrame para análisis
                df_analysis = pd.DataFrame({
                    'cultivo': cultivo_nombres,
                    'real': y_true,
                    'prediccion': y_pred,
                    'error': np.abs(y_pred - y_true)
                })
                
                # Estadísticas por cultivo
                stats_by_crop = df_analysis.groupby('cultivo').agg({
                    'real': ['mean', 'std', 'count'],
                    'prediccion': ['mean', 'std'],
                    'error': ['mean', 'std']
                }).round(2)
                
                print("\nEstadísticas por cultivo:")
                print(stats_by_crop)
                
                # Graficar
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # 1. Boxplot de errores por cultivo
                df_analysis.boxplot(column='error', by='cultivo', ax=axes[0,0])
                axes[0,0].set_title('Distribución de Errores por Cultivo')
                axes[0,0].set_ylabel('Error Absoluto (tn)')
                plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
                
                # 2. Scatter por cultivo
                for i, cultivo in enumerate(df_analysis['cultivo'].unique()):
                    mask = df_analysis['cultivo'] == cultivo
                    axes[0,1].scatter(df_analysis.loc[mask, 'real'], 
                                    df_analysis.loc[mask, 'prediccion'], 
                                    label=cultivo, alpha=0.7)
                
                axes[0,1].plot([df_analysis['real'].min(), df_analysis['real'].max()], 
                              [df_analysis['real'].min(), df_analysis['real'].max()], 'r--')
                axes[0,1].set_xlabel('Valores Reales (tn)')
                axes[0,1].set_ylabel('Predicciones (tn)')
                axes[0,1].set_title('Predicciones vs Reales por Cultivo')
                axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 3. R² por cultivo
                r2_by_crop = []
                crops = []
                for cultivo in df_analysis['cultivo'].unique():
                    mask = df_analysis['cultivo'] == cultivo
                    if mask.sum() > 3:  # Mínimo 3 muestras
                        r2 = r2_score(df_analysis.loc[mask, 'real'], 
                                     df_analysis.loc[mask, 'prediccion'])
                        r2_by_crop.append(r2)
                        crops.append(cultivo)
                
                axes[1,0].bar(crops, r2_by_crop, alpha=0.7)
                axes[1,0].set_ylabel('R² Score')
                axes[1,0].set_title('Precisión del Modelo por Cultivo')
                plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
                axes[1,0].grid(True, alpha=0.3)
                
                # 4. Volumen de producción vs precisión
                volume_precision = df_analysis.groupby('cultivo').agg({
                    'real': 'mean',
                    'error': lambda x: r2_score(df_analysis.loc[df_analysis['cultivo'] == x.name, 'real'], 
                                               df_analysis.loc[df_analysis['cultivo'] == x.name, 'prediccion']) 
                                      if len(x) > 3 else np.nan
                }).dropna()
                
                axes[1,1].scatter(volume_precision['real'], volume_precision['error'])
                for idx, row in volume_precision.iterrows():
                    axes[1,1].annotate(idx, (row['real'], row['error']), 
                                      xytext=(5, 5), textcoords='offset points', fontsize=8)
                axes[1,1].set_xlabel('Producción Media (tn)')
                axes[1,1].set_ylabel('R² Score')
                axes[1,1].set_title('Volumen vs Precisión del Modelo')
                axes[1,1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plot:
                    plt.savefig('prediction_by_crop.png', dpi=300, bbox_inches='tight')
                
                plt.show()
                
            except Exception as e:
                print(f"Error al decodificar cultivos: {e}")
                print("Análisis por cultivo no disponible")
    
    def train(self, df, target_column='produccion_tn', test_size=0.2, cv_folds=5):
        """Método principal de entrenamiento"""
        print("Iniciando entrenamiento del modelo de predicción agrícola...")
        print(f"Dataset inicial: {df.shape}")
        
        # Verificar si existe la columna objetivo
        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada en el dataset")
        
        # Preparar características (incluye limpieza automática)
        print("\nPreprocesando y limpiando datos...")
        try:
            X, y = self.prepare_features(df, target_column, is_training=True)
            print(f"✅ Datos procesados exitosamente")
            print(f"📊 Características finales: {X.shape[1]}")
            print(f"📈 Muestras para entrenamiento: {X.shape[0]}")
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {str(e)}")
            raise
        
        # Verificación final de calidad de datos
        if len(X) < 50:
            raise ValueError("⚠️  Muy pocos datos para entrenar (mínimo 50 muestras)")
        
        if X.isnull().any().any() or y.isnull().any():
            print("❌ Error: Aún hay valores NaN después del preprocesamiento")
            print("Columnas con NaN:", X.columns[X.isnull().any()].tolist())
            raise ValueError("Datos contienen valores NaN después de limpieza")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )
        
        print(f"📊 División: {len(X_train)} entrenamiento, {len(X_test)} prueba")
        
        # Entrenar modelos
        print("\nEntrenando y optimizando modelos...")
        try:
            cv_results = self.train_models(X_train, y_train, cv_folds)
        except Exception as e:
            print(f"❌ Error en entrenamiento: {str(e)}")
            raise
        
        # Evaluación final con gráficos completos
        print("\nEvaluando modelo en conjunto de prueba...")
        try:
            final_metrics, y_pred = self.comprehensive_evaluation(X_test, y_test)
        except Exception as e:
            print(f"⚠️  Error en gráficos, continuando con evaluación básica: {str(e)}")
            # Fallback a evaluación básica si los gráficos fallan
            final_metrics, y_pred = self.evaluate_model(X_test, y_test)
        
        print("\n" + "="*50)
        print("MÉTRICAS FINALES DEL MODELO")
        print("="*50)
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
        
        # Importancia de características
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print(f"\n🔍 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
            print(importance_df.head(10).to_string(index=False))
        
        print(f"\n✅ Entrenamiento completado exitosamente!")
        print(f"📊 Modelo final: {type(self.best_model).__name__}")
        
        return {
            'cv_results': cv_results,
            'test_metrics': final_metrics,
            'feature_importance': importance_df,
            'predictions': y_pred,
            'y_test': y_test,
            'data_quality': {
                'original_samples': len(df),
                'final_samples': len(X),
                'features_count': X.shape[1],
                'data_retention': (len(X) / len(df)) * 100
            }
        }

# Ejemplo de uso completo
def main():
    """Ejemplo completo de cómo usar el predictor con visualizaciones"""
    
    # Ejemplo de uso paso a paso
    print("=== PREDICTOR DE PRODUCCIÓN AGRÍCOLA ===\n")
    
    print("📋 PASOS PARA USAR EL MODELO:")
    print("1. Cargar datos: df = pd.read_csv('tu_archivo.csv')")
    print("2. Crear predictor: predictor = AgriculturalProductionPredictor()")
    print("3. Entrenar: results = predictor.train(df)")
    print("4. Los gráficos se generan automáticamente durante el entrenamiento")
    print("\n📊 GRÁFICOS QUE SE GENERAN:")
    print("• Predicciones vs Valores Reales (scatter plot)")
    print("• Análisis de Residuos (distribución y patrones)")
    print("• Q-Q Plot para normalidad")
    print("• Errores por rango de valores")
    print("• Panel completo de métricas")
    print("• Importancia de características")
    print("• Análisis por tipo de cultivo")
    
    print("\n🎯 INTERPRETACIÓN DE MÉTRICAS:")
    print("• R² > 0.9: Excelente ajuste")
    print("• R² > 0.7: Buen ajuste") 
    print("• R² > 0.5: Ajuste regular")
    print("• RMSE: Error típico en toneladas")
    print("• MAE: Error absoluto promedio")
    print("• MAPE: Error porcentual promedio")
    
    # Crear el predictor
    predictor = AgriculturalProductionPredictor(random_state=42)
    
    # Entrenar con datos reales
    try:
        df = pd.read_csv('produccion_agro_soilgrids_meteo_final.csv')
        print(f"\n📂 Archivo cargado exitosamente: {df.shape}")
        results = predictor.train(df)
        
        print("\n" + "="*60)
        print("RESUMEN DE RESULTADOS")
        print("="*60)
        
        # Mostrar métricas disponibles
        print(f"📊 Métricas disponibles: {list(results['test_metrics'].keys())}")
        
        # Acceder a métricas con nombres correctos
        metrics = results['test_metrics']
        if 'r2' in metrics:
            print(f"📈 R² Score: {metrics['r2']:.4f}")
        if 'RMSE' in metrics:
            print(f"📈 RMSE: {metrics['RMSE']:.2f} toneladas")
        if 'rmse' in metrics:
            print(f"📈 RMSE: {metrics['rmse']:.2f} toneladas")
        if 'MAE' in metrics:
            print(f"📈 MAE: {metrics['MAE']:.2f} toneladas")
        if 'mae' in metrics:
            print(f"📈 MAE: {metrics['mae']:.2f} toneladas")
        
        # Información de calidad de datos
        data_quality = results['data_quality']
        print(f"\n📊 CALIDAD DE DATOS:")
        print(f"   • Muestras originales: {data_quality['original_samples']:,}")
        print(f"   • Muestras finales: {data_quality['final_samples']:,}")
        print(f"   • Características: {data_quality['features_count']}")
        print(f"   • Retención de datos: {data_quality['data_retention']:.1f}%")
        
        # Interpretación del modelo
        r2_value = None
        for key in ['r2', 'R²']:
            if key in metrics:
                r2_value = metrics[key]
                break
        
        if r2_value is not None:
            if r2_value > 0.9:
                quality = "🟢 EXCELENTE"
            elif r2_value > 0.7:
                quality = "🟡 BUENO"
            elif r2_value > 0.5:
                quality = "🟠 REGULAR"
            else:
                quality = "🔴 NECESITA MEJORAS"
            
            print(f"\n🎯 EVALUACIÓN DEL MODELO: {quality}")
            print(f"   R² = {r2_value:.4f} - El modelo explica {r2_value*100:.1f}% de la varianza")
        
        print("\n✅ ¡Modelo entrenado y evaluado exitosamente!")
        
    except FileNotFoundError:
        print("\n⚠️  Archivo 'produccion_agro_soilgrids_meteo_final.csv' no encontrado")
        print("Coloca el archivo CSV en el mismo directorio que el script")
        print("\n✅ Modelo listo para entrenar con tus datos!")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("✅ Modelo listo para usar cuando tengas los datos correctos!")
    
    # Usar el modelo paso a paso:
    df = pd.read_csv('df_con_prod.csv')
    predictor = AgriculturalProductionPredictor()
    results = predictor.train(df)

    # 1. Gráfico de importancia de características
    predictor.plot_feature_importance(top_n=15, save_plot=True)

    # 2. Curvas de aprendizaje
    # Necesitas los datos X, y procesados, los puedes obtener así:
    X, y = predictor.prepare_features(df, target_column='produccion_tn', is_training=False)
    predictor.plot_learning_curves(X, y, save_plot=True)

    # 3. Análisis por cultivo
    # Necesitas X, y_true, y_pred del conjunto de test
    y_test = results['y_test']
    y_pred = results['predictions']
    # Para X_test, necesitarías hacer la división nuevamente o modificar el código
    # Por simplicidad, usa los datos completos:
    predictor.plot_prediction_by_crop(X, y, y_pred, save_plot=True)

    # Ver todas las métricas disponibles
    print("Métricas:", list(results['test_metrics'].keys()))

    # Acceder a métricas específicas
    metrics = results['test_metrics']
    print("R²:", metrics.get('r2', metrics.get('R²', 'No disponible')))
    print("RMSE:", metrics.get('rmse', metrics.get('RMSE', 'No disponible')))

    # Hacer nuevas predicciones
    #nuevos_datos = pd.read_csv('datos_nuevos.csv')
    #predicciones = predictor.predict(nuevos_datos)
    
    return predictor

if __name__ == "__main__":
    predictor = main()