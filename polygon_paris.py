import folium
import requests
import pandas as pd
import shapely.geometry as shape

# Url récupérant les données géographiques de Paris
url_paris= "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-france-arrondissement-departemental/exports/geojson?select=%2A&where=dep_name%3D%22Paris%22&limit=-1&lang=fr&timezone=UTC&use_labels=false&compressed=false&epsg=4326"

def retrieve_polygon_paris(url_paris=url_paris):
    """
    Récupère le polygone de Paris à partir de l'API OpenData"""
    try:
        response_paris = requests.get(url_paris) # requête GET pour récupérer les données géographiques de Paris
        data_paris = response_paris.json() # conversion de la réponse en format JSON

        features = data_paris.get('features', [])  # récupération de la liste des features (polygones) de Paris

        for f in features:
            polygon_paris = f['geometry'] # récupération de la géométrie du polygone de Paris
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None
    carte = folium.Map(location=[48.8566, 2.3522], zoom_start=12) # création d'une carte centrée sur Paris
    folium.GeoJson(polygon_paris, name="geojson").add_to(carte) # ajout du polygone de Paris à la carte

    shape_polygon = shape.shape(polygon_paris) # création d'un objet shapely à partir du polygone de Paris

    return carte, polygon_paris, shape_polygon




