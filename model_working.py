from robbery_model import RobberyPredictionModel
from training_routines import train_with_sqlite, train_with_real_data, demo_with_simulated_data
import os

if __name__ == "__main__":
    print("🚔 MODELO DE PREDICCIÓN DE ROBOS EN MEDELLÍN 🚔\n")

    modelos = None
    model = None

    # Opciones:
    #   - 'csv': usar archivo CSV
    #   - 'sqlite': usar base de datos SQLite
    #   - 'demo': demo con datos simulados
    data_source = 'sqlite'  # Cambiar según sea necesario

    if data_source == 'csv':
        print("Opción 1: Entrenar con archivo CSV real...")
        pkl_paths = {
            'random_forest': 'modelo_robos_medellin_random_forest.pkl',
            'neural_network': 'modelo_robos_medellin_neural_network.pkl'
        }
        modelos = {}
        modelos_cargados = {}
        for model_type, pkl_path in pkl_paths.items():
            if os.path.exists(pkl_path):
                print(f"Modelo {model_type} encontrado en {pkl_path}. Cargando modelo...")
                m = RobberyPredictionModel(model_type=model_type)
                m.load_model(pkl_path)
                modelos[model_type] = m
                modelos_cargados[model_type] = True
            else:
                modelos_cargados[model_type] = False
        if not any(modelos_cargados.values()):
            print("No se encontraron modelos guardados. Entrenando nuevos modelos...")
            modelos, resultados = train_with_real_data()
        else:
            print("\nModelos cargados correctamente desde archivos pkl.")
    elif data_source == 'sqlite':
        print("Opción 2: Entrenar con base de datos SQLite...")
        pkl_paths = {
            'random_forest': 'assets/models/modelo_robos_medellin_random_forest.pkl',
            'neural_network': 'assets/models/modelo_robos_medellin_neural_network.pkl'
        }
        modelos = {}
        modelos_cargados = {}
        for model_type, pkl_path in pkl_paths.items():
            if os.path.exists(pkl_path):
                print(f"Modelo {model_type} encontrado en {pkl_path}. Cargando modelo...")
                m = RobberyPredictionModel(model_type=model_type)
                m.load_model(pkl_path)
                modelos[model_type] = m
                modelos_cargados[model_type] = True
            else:
                modelos_cargados[model_type] = False
        if not any(modelos_cargados.values()):
            print("No se encontraron modelos guardados. Entrenando nuevos modelos...")
            modelos, resultados = train_with_sqlite(
                db_path='assets/hurto.db',
                table_name='hurtos'
            )
        else:
            print("\nModelos cargados correctamente desde archivos pkl.")
    else:
        print("Opción 3: Demo con datos simulados...")
        pkl_path = 'modelo_robos_medellin.pkl'
        if os.path.exists(pkl_path):
            print(f"Modelo encontrado en {pkl_path}. Cargando modelo...")
            model = RobberyPredictionModel()
            model.load_model(pkl_path)
        else:
            print("No se encontró modelo guardado. Entrenando uno nuevo...")
            model = demo_with_simulated_data()

    # Chequeo final
    if modelos and len(modelos) > 0:
        print("\n✅ Modelos disponibles!")
        print("\n📋 INSTRUCCIONES DE USO:")
        for k in modelos:
            print(f"- Para predecir riesgo con {k.replace('_', ' ').title()}: modelos['{k}'].predict_risk_zone(lat, lon, hora=20, modalidad='Atraco')")
        print("2. Los modelos se guardan automáticamente como 'modelo_robos_medellin_random_forest.pkl' y/o 'modelo_robos_medellin_neural_network.pkl'")
    elif model:
        print("\n✅ Modelo disponible!")
        print("\n📋 INSTRUCCIONES DE USO:")
        print("1. Para predecir riesgo: model.predict_risk_zone(lat, lon, hora=20, modalidad='Atraco')")
        print("2. El modelo se guarda automáticamente como 'modelo_robos_medellin.pkl'")
    else:
        print("\n❌ No se pudo cargar ni entrenar el modelo.")
        print("Verifica que tengas los datos correctos o las librerías instaladas.") 