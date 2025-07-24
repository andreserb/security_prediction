import osmnx as ox
import networkx as nx
import folium
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import BallTree
from osmnx import graph_to_gdfs
from robbery_model import RobberyPredictionModel
from risk_utils import get_high_risk_zones


#ENTRADA DE USUARIO (puede venir de formulario o script externo) ===
origen = (6.299703, -75.582016)
destino = (6.250917, -75.566160)


# === 1. CACHE LOCAL ===
def local_map():
  graphml_file = "assets/graphos/medellin.graphml"
  if os.path.exists(graphml_file):
      print("Cargando red vial desde caché local...")
      G = ox.load_graphml(graphml_file)
  else:
      print("Descargando red vial de Medellín...")
      G = ox.graph_from_place("Medellín, Colombia", network_type="drive")
      ox.save_graphml(G, graphml_file)
  return G

# === 2. ZONAS DE ALTO RIESGO (simuladas) ===
def danger_zones(G, zonas_peligrosas, radio_riesgo_m):
    # Obtener todos los nodos del grafo con sus coordenadas
    gdf_nodes = graph_to_gdfs(G, edges=False)
    if gdf_nodes.empty:
        return set()
    # Convertir a array de [lat, lon] para BallTree
    coords_nodes = np.array([[geom.y, geom.x] for geom in gdf_nodes.geometry])
    # Crear BallTree para búsquedas espaciales eficientes
    tree = BallTree(np.deg2rad(coords_nodes), metric='haversine')
    # Calcular radio en radianes (metros a radianes)
    radio_rad = radio_riesgo_m / 6371000  # Radio terrestre en metros
    nodos_prohibidos = set()
    for lat, lon in zonas_peligrosas:
        # Encontrar todos los nodos dentro del radio de peligro
        indices = tree.query_radius(
            np.deg2rad([[lat, lon]]),
            r=radio_rad
        )
        # Agregar IDs de nodos encontrados
        for idx in indices[0]:
            node_id = gdf_nodes.index[idx]
            nodos_prohibidos.add(node_id)
    return nodos_prohibidos

# === 3. CÁLCULO DE RUTA SEGURA ===
def calculate_route(G, nodo_origen, nodo_destino, zonas_peligrosas, radio_riesgo_m=200):
    # Identificar nodos peligrosos
    nodos_prohibidos = danger_zones(G, zonas_peligrosas, radio_riesgo_m)
    # Excluir origen/destino si están en la lista
    nodos_prohibidos = nodos_prohibidos - {nodo_origen, nodo_destino}
    # Crear copia modificable del grafo
    G_seguro = G.copy()
    # Remover conexiones peligrosas (no solo nodos)
    for nodo in nodos_prohibidos:
        if G_seguro.has_node(nodo):
            # Eliminar aristas entrantes y salientes
            edges_to_remove = list(G_seguro.in_edges(nodo)) + list(G_seguro.out_edges(nodo))
            G_seguro.remove_edges_from(edges_to_remove)
    # Buscar ruta evitando zonas peligrosas
    try:
        ruta = nx.shortest_path(
            G_seguro,
            nodo_origen,
            nodo_destino,
            weight='length',
            method='dijkstra'
        )
        return ruta
    except nx.NetworkXNoPath:
        # Fallback: intentar con grafo completo si no hay ruta segura
        try:
            print("⚠️ No hay ruta segura, usando ruta alternativa")
            return nx.shortest_path(G, nodo_origen, nodo_destino, weight='length')
        except nx.NetworkXNoPath:
            print("❌ Error crítico: No existe ruta posible")
            return []

# === 4. MAPA INTERACTIVO ===
def create_map(ruta, origen, destino, zonas_peligrosas, radio_riesgo_m):
  m = folium.Map(location=origen, zoom_start=13)
  # Marcadores
  folium.Marker(origen, tooltip="Origen", icon=folium.Icon(color='green')).add_to(m)
  folium.Marker(destino, tooltip="Destino", icon=folium.Icon(color='blue')).add_to(m)
  # Puntos de riesgo
  for lat, lon in zonas_peligrosas:
      folium.Circle(
          location=(lat, lon),
          radius=radio_riesgo_m,
          color='red',
          fill=True,
          fill_opacity=0.4,
          tooltip="Zona peligrosa"
      ).add_to(m)
  # Ruta segura
  if ruta:
      coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in ruta]
      folium.PolyLine(coords, color='blue', weight=5, tooltip="Ruta segura").add_to(m)
  # Mostrar o guardar
  m.save("assets/maps/ruta_segura.html")

# Funcion Main
if __name__ == "__main__":
    G = local_map()
    nodo_origen = ox.distance.nearest_nodes(G, X=origen[1], Y=origen[0])
    nodo_destino = ox.distance.nearest_nodes(G, X=destino[1], Y=destino[0])

    # Obtener todos los nodos del grafo como posibles puntos a evaluar
    gdf_nodes = graph_to_gdfs(G, edges=False)
    coords_nodes = [(geom.y, geom.x) for geom in gdf_nodes.geometry]

    # Cargar modelo de riesgo (random forest por defecto)
    modelo = RobberyPredictionModel(model_type='random_forest')
    modelo.load_model('assets/models/modelo_robos_medellin_random_forest.pkl')

    # Predecir zonas de alto riesgo
    print("Calculando zonas de alto riesgo en la red vial...")
    zonas_riesgo = get_high_risk_zones(modelo, coords_nodes, threshold_label='Alto')
    print(f"Nodos de alto riesgo detectados: {len(zonas_riesgo)}")

    # Calcular ruta segura
    ruta_segura = calculate_route(
        G,
        nodo_origen=nodo_origen,
        nodo_destino=nodo_destino,
        zonas_peligrosas=zonas_riesgo,
        radio_riesgo_m=200
    )
    create_map(ruta_segura, origen, destino, zonas_riesgo, radio_riesgo_m=200)
