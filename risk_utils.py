from robbery_model import RobberyPredictionModel

def get_high_risk_zones(model, coords, threshold_label='Alto'):
    """
    Devuelve las coordenadas clasificadas como alto riesgo.
    model: instancia de RobberyPredictionModel ya cargada
    coords: lista de (lat, lon)
    threshold_label: etiqueta de riesgo a considerar (por defecto 'Alto')
    """
    zonas_peligrosas = []
    for lat, lon in coords:
        pred = model.predict_risk_zone(lat, lon)
        if pred['nivel_riesgo'] == threshold_label:
            zonas_peligrosas.append((lat, lon))
    return zonas_peligrosas 