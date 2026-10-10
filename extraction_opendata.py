from datetime import date, timedelta
import requests
import pandas as pd
from shapely.geometry import Point

today = date.today()
date_passee = (today - timedelta(days=365)).strftime("%Y-%m-%d")  # Date d'il y a un an
location_name = "Paris"
lang = "fr"

def retrieve_opendata(location_name=location_name,lang=lang,lastdate_end=date_passee):
    url= "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json?select=uid%2C%20slug%2C%20title_fr%2C%20description_fr%2C%20longdescription_fr%2C%20conditions_fr%2C%20keywords_fr%2C%20lastdate_begin%2C%20lastdate_end%2C%20%20location_coordinates%2C%20location_name%2C%20location_address%2C%20%20location_district%2C%20location_postalcode%2C%20location_city%2C%20location_department%2C%20%20location_region%2C%20location_countrycode%2C%20%20country_fr&where=location_city%20%3D%20%22"+location_name+"%22%20AND%20lastdate_end%3E%3D%22"+date_passee+"T00%3A00%3A00%2B00%3A00%22&limit=-1&lang="+lang+"&timezone=UTC&use_labels=false&compressed=false&epsg=4326"
    try:
        response = requests.get(url)
        df = pd.DataFrame(response.json())
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None
        
    return df

def extract_gps_coordinates(df:pd.DataFrame(),column="location_coordinates"):
    try:
        df['latitude'] = df[column].apply(lambda x: x['lat']).astype('Float64')
        df['longitude'] = df[column].apply(lambda x: x['lon']).astype('Float64')
        df = df.drop(column, axis=1)
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        print("Vérifiez les données")
        return None
    return df

def coordinates_in_frontiere(df:pd.DataFrame(),frontiere):
    """
    Sert à garder les coordonnées gps qui sont dans un polygone donné (frontière)
    """
    try:
        valid_mask = ((df['latitude'].notna())|(df['longitude'].notna()))
        rejected_coordinates = df.loc[~valid_mask].copy()
        
        candidates = df.loc[valid_mask].copy()
        
        points = candidates.apply(
        lambda row: Point(row["longitude"], row["latitude"]),
        axis=1,
        )

        in_frontiere_mask = points.map(frontiere.covers)

        events_in_frontiere = candidates.loc[in_frontiere_mask].copy()
        events_outside_frontiere = candidates.loc[~in_frontiere_mask].copy()
 

    except Exception as e:
        print(f"Erreur de l'exécution de la fonction : {e}")
        print("Vérifier si les colonnes latitude et longitude existent bien \n Vérifier la nature de la frontière.")
    
    return events_in_frontiere, rejected_coordinates, events_outside_frontiere