import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import sqlite3

class RobberyPredictionModel:
    def __init__(self, model_type='random_forest'):
        """
        Modelo para predicción de robos en Medellín
        Args:
            model_type: 'random_forest' o 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_zones = None

    def load_and_preprocess_data(self, file_path=None, csv_data=None):
        if csv_data is not None:
            df = csv_data.copy()
        elif file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
        else:
            raise ValueError("Debe proporcionar file_path o csv_data")

        print(f"Datos cargados: {len(df)} registros")

        # Convertir fecha
        if 'fecha_hecho' in df.columns:
            df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'])
            df['hora'] = df['fecha_hecho'].dt.hour
            df['dia_semana'] = df['fecha_hecho'].dt.dayofweek
            df['mes'] = df['fecha_hecho'].dt.month
            df['anio'] = df['fecha_hecho'].dt.year

        # Limpiar coordenadas - manejar diferentes formatos
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        df = df.dropna(subset=['latitud', 'longitud'])

        # Filtrar coordenadas válidas para Medellín
        df = df[
            (df['latitud'] >= 6.1) & (df['latitud'] <= 6.4) &
            (df['longitud'] >= -75.7) & (df['longitud'] <= -75.4)
        ]

        # Manejar valores faltantes en columnas importantes
        for col in ['sexo', 'modalidad', 'medio_transporte', 'lugar', 'arma_medio', 'nombre_barrio']:
            if col in df.columns:
                df[col] = df[col].fillna('Sin_dato')

        # Convertir edad a numérico y manejar valores faltantes
        if 'edad' in df.columns:
            df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
            df['edad'] = df['edad'].fillna(df['edad'].median())
            df['edad'] = df['edad'].apply(lambda x: x if 10 <= x <= 100 else 30)  # Filtrar edades inválidas
        else:
            df['edad'] = 30  # Valor predeterminado si no hay columna edad

        # Crear zonas de riesgo basadas en clustering geográfico
        self.create_risk_zones(df)

        # Feature engineering
        df = self.create_features(df)

        print(f"Datos procesados: {len(df)} registros")
        return df

    def create_risk_zones(self, df, n_zones=4):
        coords = df[['latitud', 'longitud']].values
        kmeans = KMeans(n_clusters=n_zones, random_state=42)
        df['zona_riesgo'] = kmeans.fit_predict(coords)
        zone_stats = df.groupby('zona_riesgo').agg({
            'latitud': 'mean',
            'longitud': 'mean'
        }).reset_index()
        incident_counts = df['zona_riesgo'].value_counts().reset_index()
        incident_counts.columns = ['zona_riesgo', 'incidentes']
        zone_stats = zone_stats.merge(incident_counts, on='zona_riesgo')
        zone_stats = zone_stats.sort_values('incidentes', ascending=False)
        zone_mapping = {
            zone_stats.iloc[0]['zona_riesgo']: 3,
            zone_stats.iloc[1]['zona_riesgo']: 2,
            zone_stats.iloc[2]['zona_riesgo']: 1,
            zone_stats.iloc[3]['zona_riesgo']: 0
        }
        df['nivel_riesgo'] = df['zona_riesgo'].map(zone_mapping)
        self.target_zones = zone_mapping
        print("\nDistribución de zonas de riesgo:")
        print(df['nivel_riesgo'].value_counts().sort_index())
        return df

    def create_features(self, df):
        if 'hora' in df.columns:
            df['es_noche'] = ((df['hora'] >= 18) | (df['hora'] <= 6)).astype(int)
            df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
            df['franja_horaria'] = pd.cut(df['hora'],
                                        bins=[0, 6, 12, 18, 24],
                                        labels=['madrugada', 'mañana', 'tarde', 'noche'],
                                        include_lowest=True,
                                        ordered=True)
        categorical_features = ['sexo', 'modalidad', 'medio_transporte', 'lugar',
                            'arma_medio', 'nombre_barrio', 'franja_horaria']
        categorical_features = [col for col in categorical_features if col in df.columns]
        for feature in categorical_features:
            if pd.api.types.is_categorical_dtype(df[feature]):
                df[feature] = df[feature].astype(str)
            df[feature] = df[feature].fillna('Sin_dato')
            value_counts = df[feature].value_counts()
            frequent_categories = value_counts[value_counts >= 10].index
            df[f'{feature}_grouped'] = df[feature].apply(
                lambda x: x if x in frequent_categories else 'Otros'
            )
        return df

    def prepare_features(self, df):
        numeric_features = ['latitud', 'longitud', 'edad', 'hora', 'dia_semana',
                          'mes', 'es_noche', 'es_fin_semana']
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [col for col in df.columns if col.endswith('_grouped')]
        categorical_features = [f for f in categorical_features if f in df.columns]
        X_numeric = df[numeric_features].fillna(0)
        X_categorical = pd.DataFrame()
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                X_categorical[feature] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                unique_values = self.label_encoders[feature].classes_
                df[feature] = df[feature].apply(
                    lambda x: x if x in unique_values else unique_values[0]
                )
                X_categorical[feature] = self.label_encoders[feature].transform(df[feature])
        if len(X_categorical.columns) > 0:
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X = X_numeric
        self.feature_columns = X.columns.tolist()
        return X.values

    def build_neural_model(self, input_shape, num_classes=4):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está disponible")
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, df):
        X = self.prepare_features(df)
        y = df['nivel_riesgo'].values
        if len(X) != len(y):
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            print(f"Ajustado a tamaño común: {min_length} muestras")
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        if self.model_type == 'random_forest':
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5]
            }
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
            print("Entrenando Random Forest con Grid Search...")
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Mejores parámetros: {grid_search.best_params_}")
        elif self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            y_train_cat = to_categorical(y_train, 4)
            y_test_cat = to_categorical(y_test, 4)
            self.model = self.build_neural_model(X_train.shape[1])
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
            ]
            print("Entrenando Red Neuronal...")
            history = self.model.fit(
                X_train, y_train_cat,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            y_test = y_test_cat
        self.evaluate_model(X_test, y_test)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test, y_test):
        if self.model_type == 'random_forest':
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nReporte de clasificación:")
            print(classification_report(y_test, y_pred,
                                      target_names=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']))
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nCaracterísticas más importantes:")
                print(feature_importance.head(10))
        elif self.model_type == 'neural_network':
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Loss: {loss:.4f}")

    def predict_risk_zone(self, lat, lon, **kwargs):
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        data = {
            'latitud': lat,
            'longitud': lon,
            'edad': kwargs.get('edad', 30),
            'hora': kwargs.get('hora', 12),
            'dia_semana': kwargs.get('dia_semana', 1),
            'mes': kwargs.get('mes', 6),
            'es_noche': kwargs.get('es_noche', 0),
            'es_fin_semana': kwargs.get('es_fin_semana', 0)
        }
        for feature in self.label_encoders.keys():
            base_feature = feature.replace('_grouped', '')
            data[feature] = kwargs.get(base_feature, 'Sin_dato')
        df_pred = pd.DataFrame([data])
        for feature, encoder in self.label_encoders.items():
            if feature in df_pred.columns:
                try:
                    df_pred[feature] = encoder.transform(df_pred[feature])
                except ValueError:
                    df_pred[feature] = encoder.transform([encoder.classes_[0]])[0]
        X_pred = df_pred[self.feature_columns].fillna(0).values
        X_pred_scaled = self.scaler.transform(X_pred)
        if self.model_type == 'random_forest':
            prediction = self.model.predict(X_pred_scaled)[0]
            probabilities = self.model.predict_proba(X_pred_scaled)[0]
        else:
            prediction = np.argmax(self.model.predict(X_pred_scaled, verbose=0), axis=1)[0]
            probabilities = self.model.predict(X_pred_scaled, verbose=0)[0]
        risk_levels = ['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']
        return {
            'nivel_riesgo': risk_levels[prediction],
            'probabilidades': dict(zip(risk_levels, probabilities))
        }

    def save_model(self, filepath='robbery_model.pkl'):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'target_zones': self.target_zones
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath='robbery_model.pkl'):
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.target_zones = model_data.get('target_zones')
            print(f"Modelo cargado desde: {filepath}")
            return True
        return False

    def load_data_from_sqlite(self, db_path, table_name):
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"Datos cargados desde SQLite: {len(df)} registros")
            return df
        except sqlite3.Error as e:
            print(f"Error de SQLite: {e}")
            return None
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return None 