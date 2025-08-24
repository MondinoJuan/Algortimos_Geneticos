import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

class CropProductionRFModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.climate_features = None
        self.numeric_cols = ['anio','organic_carbon','ph','clay','silt','sand',
                             'superficie_sembrada_ha','produccion_tn']
        self.categorical_cols = ['cultivo_nombre','departamento_nombre']

    def load_and_clean_data(self, csv_path):
        print("📊 Cargando datos...")
        df = pd.read_csv(csv_path)

        print(f"Forma original: {df.shape}")
        print(f"Cultivos únicos: {df['cultivo_nombre'].nunique()}")
        print(f"Departamentos únicos: {df['departamento_nombre'].nunique()}")
        print(f"Rango de años: {df['anio'].min()} - {df['anio'].max()}")

        df_clean = self.clean_missing_values(df)
        print(f"Datos después de limpieza: {df_clean.shape}")
        return df_clean

    def clean_missing_values(self, df):
        df = df.copy()
        df[self.numeric_cols] = df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['produccion_tn'])
        df = df[df['produccion_tn'] > 0]
        imputer = SimpleImputer(strategy="median")
        df[self.numeric_cols] = imputer.fit_transform(df[self.numeric_cols])
        return df

    def process_climate_arrays(self, df):
        print("🌡️ Procesando datos climáticos...")
        climate_cols = {
            'temperatura_media_C': 'temp',
            'humedad_relativa_%': 'hum', 
            'velocidad_viento_m_s': 'viento_ms',
            'velocidad_viento_km_h': 'viento_kmh',
            'precipitacion_mm_mes': 'precip'
        }
        df_processed = df.copy()
        climate_features_created = []

        def safe_parse(x):
            if pd.isna(x) or str(x).strip() in ['SD','N/A','',' ']:
                return None
            try:
                parsed = ast.literal_eval(x) if isinstance(x, str) else x
                if isinstance(parsed, list):
                    vals = [float(v) for v in parsed if isinstance(v,(int,float)) and not np.isnan(v)]
                    return vals if vals else None
            except:
                return None
            return None

        for original_col, short_name in climate_cols.items():
            if original_col in df_processed.columns:
                print(f"  Procesando {original_col}...")
                parsed_arrays = df_processed[original_col].map(safe_parse)
                stats = {
                    f"{short_name}_mean": np.nanmean,
                    f"{short_name}_std": np.nanstd,
                    f"{short_name}_min": np.nanmin,
                    f"{short_name}_max": np.nanmax,
                    f"{short_name}_median": np.nanmedian
                }
                for fname, func in stats.items():
                    df_processed[fname] = parsed_arrays.apply(lambda x: func(x) if x else np.nan)
                    climate_features_created.append(fname)
                # rango
                df_processed[f"{short_name}_range"] = parsed_arrays.apply(
                    lambda x: (np.nanmax(x)-np.nanmin(x)) if x and len(x)>1 else 0.0)
                climate_features_created.append(f"{short_name}_range")
                df_processed = df_processed.drop(columns=[original_col])

        for feature in climate_features_created:
            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())

        self.climate_features = climate_features_created
        print(f"  Características climáticas creadas: {len(climate_features_created)}")
        return df_processed

    def encode_categorical_variables(self, df):
        print("🏷️ Codificando variables categóricas...")
        df_encoded = df.copy()
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                print(f"  Codificando {col}: {df_encoded[col].nunique()} categorías únicas")
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        return df_encoded

    def prepare_features_and_target(self, df):
        base_features = ['cultivo_nombre','anio','departamento_nombre','organic_carbon','ph','clay','silt','sand','superficie_sembrada_ha']
        all_features = base_features + self.climate_features
        available_features = [col for col in all_features if col in df.columns]
        missing_features = [col for col in all_features if col not in df.columns]
        if missing_features:
            print(f"⚠️ Características faltantes: {missing_features}")
        print(f"📋 Usando {len(available_features)} características para el modelo")
        X = df[available_features].copy()
        y = df['produccion_tn'].copy()
        self.feature_names = available_features
        return X, y

    def train_model(self, X, y, test_size=0.15, optimize=True):
        print("\n🚀 Entrenando Random Forest Regressor...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True)
        if optimize:
            print("🔧 Optimizando hiperparámetros...")
            self.model = self.optimize_hyperparameters(X_train, y_train)
        else:
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt',
                random_state=42, n_jobs=-1, oob_score=True
            )
            self.model.fit(X_train, y_train)
        results = self.evaluate_model(X_train, X_test, y_train, y_test)
        return results, X_train, X_test, y_train, y_test

    def optimize_hyperparameters(self, X_train, y_train):
        param_dist = {
            'n_estimators': [100, 200, 500],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
        search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=20, cv=5,
            scoring="neg_mean_squared_error", n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        print(f"✅ Mejores parámetros: {search.best_params_}")
        print(f"✅ Mejor score CV: {-search.best_score_:.2f} RMSE")
        return search.best_estimator_

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_pred': train_pred,
            'test_pred': test_pred,
            'oob_score': getattr(self.model, 'oob_score_', None)
        }
        print(f"\n📊 RESULTADOS:")
        print(f"R² Entrenamiento: {metrics['train_r2']:.4f}")
        print(f"R² Test: {metrics['test_r2']:.4f}")
        print(f"RMSE Test: {metrics['test_rmse']:.2f}")
        return metrics

    def plot_feature_importance(self, top_n=20):
        if self.model is None:
            print("❌ El modelo no ha sido entrenado aún")
            return
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(12,8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n), palette='Blues_r')
        plt.title(f'Top {top_n} características más importantes')
        plt.tight_layout()
        plt.show()

    def save_model(self, path="rf_model.pkl"):
        joblib.dump({"model": self.model, "encoders": self.label_encoders,
                     "features": self.feature_names}, path)

    def load_model(self, path="rf_model.pkl"):
        saved = joblib.load(path)
        self.model = saved["model"]
        self.label_encoders = saved["encoders"]
        self.feature_names = saved["features"]

    def predict_new_samples(self, new_data):
        if self.model is None:
            raise ValueError("❌ El modelo no ha sido entrenado aún")
        new_data_processed = self.process_climate_arrays(new_data)
        for col, encoder in self.label_encoders.items():
            if col in new_data_processed.columns:
                new_data_processed[col] = encoder.transform(new_data_processed[col].astype(str))
        for feature in self.feature_names:
            if feature not in new_data_processed.columns:
                new_data_processed[feature] = 0
        X_new = new_data_processed[self.feature_names]
        return self.model.predict(X_new)

def main():
    model = CropProductionRFModel()
    csv_path = "df_con_prod.csv"
    try:
        df = model.load_and_clean_data(csv_path)
        df_processed = model.process_climate_arrays(df)
        df_encoded = model.encode_categorical_variables(df_processed)
        X, y = model.prepare_features_and_target(df_encoded)
        results, X_train, X_test, y_train, y_test = model.train_model(X, y, optimize=True)
        model.plot_feature_importance(top_n=20)
        print(f"\n✅ ¡Modelo Random Forest entrenado!")
        print(f"📈 R² test: {results['test_r2']:.4f}")
        print(f"📉 RMSE test: {results['test_rmse']:.2f}")
        model.save_model()
        return model, results
    except FileNotFoundError:
        print("❌ Archivo no encontrado.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback; traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, results = main()
