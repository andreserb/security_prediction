from robbery_model import RobberyPredictionModel
import pandas as pd
import numpy as np

# ================= FUNCIONES AUXILIARES =================

def train_with_sqlite(db_path='assets/hurto.db', table_name='hurtos'):
    print("=== ENTRENAMIENTO CON DATOS REALES (SQLite) ===")
    modelos = {}
    resultados = {}
    for model_type in ['random_forest', 'neural_network']:
        if model_type == 'neural_network' and not RobberyPredictionModel.TENSORFLOW_AVAILABLE:
            print("TensorFlow no disponible. Se omite el entrenamiento de la red neuronal.")
            continue
        print(f"\n--- Entrenando modelo: {model_type.replace('_', ' ').title()} ---")
        model = RobberyPredictionModel(model_type=model_type)
        try:
            print(f"Cargando datos desde {db_path}, tabla {table_name}...")
            df = model.load_data_from_sqlite(db_path, table_name)
            if df is None or len(df) == 0:
                print("No se pudieron cargar datos desde SQLite")
                continue
            print("Columnas disponibles:", df.columns.tolist())
            df_processed = model.load_and_preprocess_data(csv_data=df)
            print("\nEntrenando modelo...")
            X_train, X_test, y_train, y_test = model.train_model(df_processed)
            model_filename = f"modelo_robos_medellin_{model_type}.pkl"
            model.save_model(model_filename)
            print("\n=== ANÁLISIS DE ZONAS DE RIESGO ===")
            zone_analysis = df_processed.groupby('nivel_riesgo').agg({
                'latitud': 'mean',
                'longitud': 'mean',
                'nombre_barrio': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(4).reset_index()
            zone_labels = {0: 'Bajo', 1: 'Medio-Bajo', 2: 'Medio-Alto', 3: 'Alto'}
            zone_analysis['nivel_riesgo'] = zone_analysis['nivel_riesgo'].map(zone_labels)
            print(zone_analysis)
            print("\n=== EJEMPLOS DE PREDICCIÓN ===")
            test_locations = [
                (6.2442, -75.5812, "Centro de Medellín"),
                (6.2077, -75.5761, "El Poblado"),
                (6.2518, -75.5636, "La Candelaria"),
                (6.2304, -75.5916, "Laureles")
            ]
            for lat, lon, nombre in test_locations:
                resultado = model.predict_risk_zone(
                    lat=lat, lon=lon,
                    hora=20,  # 8 PM
                    edad=30,
                    modalidad="Atraco"
                )
                print(f"\n📍 {nombre} ({lat}, {lon}) a las 8 PM:")
                print(f"   Nivel de riesgo: {resultado['nivel_riesgo']}")
                max_prob = max(resultado['probabilidades'].values())
                print(f"   Probabilidad más alta: {max_prob:.3f}")
            modelos[model_type] = model
            resultados[model_type] = {
                'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test
            }
        except Exception as e:
            print(f"Error durante el entrenamiento con {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    return modelos, resultados

def train_with_real_data():
    print("=== ENTRENAMIENTO CON DATOS REALES ===")
    modelos = {}
    resultados = {}
    for model_type in ['random_forest', 'neural_network']:
        if model_type == 'neural_network' and not RobberyPredictionModel.TENSORFLOW_AVAILABLE:
            print("TensorFlow no disponible. Se omite el entrenamiento de la red neuronal.")
            continue
        print(f"\n--- Entrenando modelo: {model_type.replace('_', ' ').title()} ---")
        model = RobberyPredictionModel(model_type=model_type)
        try:
            print("Cargando datos desde hurtos.csv...")
            df = pd.read_csv('hurtos.csv', delimiter=';', encoding='utf-8')
            print(f"Datos cargados: {len(df)} registros")
            print("Columnas disponibles:", df.columns.tolist())
            df_processed = model.load_and_preprocess_data(csv_data=df)
            print("\nEntrenando modelo...")
            X_train, X_test, y_train, y_test = model.train_model(df_processed)
            model_filename = f"modelo_robos_medellin_{model_type}.pkl"
            model.save_model(model_filename)
            print("\n=== ANÁLISIS DE ZONAS DE RIESGO ===")
            zone_analysis = df_processed.groupby('nivel_riesgo').agg({
                'latitud': 'mean',
                'longitud': 'mean',
                'nombre_barrio': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(4).reset_index()
            zone_labels = {0: 'Bajo', 1: 'Medio-Bajo', 2: 'Medio-Alto', 3: 'Alto'}
            zone_analysis['nivel_riesgo'] = zone_analysis['nivel_riesgo'].map(zone_labels)
            print(zone_analysis)
            print("\n=== EJEMPLOS DE PREDICCIÓN ===")
            test_locations = [
                (6.2442, -75.5812, "Centro de Medellín"),
                (6.2077, -75.5761, "El Poblado"),
                (6.2518, -75.5636, "La Candelaria"),
                (6.2304, -75.5916, "Laureles")
            ]
            for lat, lon, nombre in test_locations:
                resultado = model.predict_risk_zone(
                    lat=lat, lon=lon,
                    hora=20,  # 8 PM
                    edad=30,
                    modalidad="Atraco"
                )
                print(f"\n📍 {nombre} ({lat}, {lon}) a las 8 PM:")
                print(f"   Nivel de riesgo: {resultado['nivel_riesgo']}")
                max_prob = max(resultado['probabilidades'].values())
                print(f"   Probabilidad más alta: {max_prob:.3f}")
            modelos[model_type] = model
            resultados[model_type] = {
                'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test
            }
        except FileNotFoundError:
            print("Error: No se encontró el archivo 'hurtos.csv'")
            print("Asegúrate de que el archivo esté en el directorio actual")
            continue
        except Exception as e:
            print(f"Error durante el entrenamiento con {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    return modelos, resultados

def demo_with_simulated_data():
    print("=== DEMO CON DATOS SIMULADOS ===")
    model = RobberyPredictionModel(model_type='random_forest')
    np.random.seed(42)
    n_samples = 1000
    zones = [
        (6.2442, -75.5812),  # Centro
        (6.2077, -75.5761),  # Poblado
        (6.2518, -75.5636),  # Candelaria
        (6.2304, -75.5916)   # Laureles
    ]
    data = []
    for i in range(n_samples):
        zone_lat, zone_lon = zones[np.random.randint(0, len(zones))]
        lat = zone_lat + np.random.normal(0, 0.01)
        lon = zone_lon + np.random.normal(0, 0.01)
        if np.random.random() < 0.3:
            hora = np.random.randint(18, 24)
        else:
            hora = np.random.randint(6, 18)
        fecha_base = pd.Timestamp('2017-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        fecha_str = fecha_base.strftime('%Y-%m-%dT') + f"{hora:02d}:00:00.000-05:00"
        data.append({
            'fecha_hecho': fecha_str,
            'latitud': lat,
            'longitud': lon,
            'sexo': np.random.choice(['Hombre', 'Mujer']),
            'edad': np.random.randint(18, 70),
            'modalidad': np.random.choice(['Atraco', 'Descuido', 'Engaño']),
            'medio_transporte': np.random.choice(['Caminata', 'Taxi', 'Motocicleta', 'Sin dato']),
            'lugar': np.random.choice(['Vía pública', 'Hospital o centro de salud']),
            'nombre_barrio': np.random.choice(['Centro', 'Poblado', 'Candelaria', 'Laureles']),
            'arma_medio': np.random.choice(['Arma de fuego', 'Arma cortopunzante', 'No', 'Sin dato'])
        })
    df = pd.DataFrame(data)
    try:
        df_processed = model.load_and_preprocess_data(csv_data=df)
        model.train_model(df_processed)
        resultado = model.predict_risk_zone(
            lat=6.2442, lon=-75.5812,
            hora=20, edad=25, modalidad="Atraco"
        )
        print(f"\n📍 Predicción para Centro de Medellín:")
        print(f"   Nivel de riesgo: {resultado['nivel_riesgo']}")
        print(f"   Probabilidades: {resultado['probabilidades']}")
        return model
    except Exception as e:
        print(f"Error en demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 