from datetime import date, timedelta
import requests
import pandas as pd
from shapely.geometry import Point
import folium

today = date.today()
date_passee = (today - timedelta(days=365)).strftime("%Y-%m-%d")  # Date d'il y a un an
location_name = "Paris"
lang = "fr"
location_code = "location_postalcode"

def retrieve_opendata(location_name=location_name,lang=lang,lastdate_end=date_passee):
    url= "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json?select=uid%2C%20slug%2C%20title_fr%2C%20description_fr%2C%20longdescription_fr%2C%20conditions_fr%2C%20keywords_fr%2C%20lastdate_begin%2C%20lastdate_end%2C%20%20location_coordinates%2C%20location_name%2C%20location_address%2C%20%20location_district%2C%20"+location_code+"%2C%20location_city%2C%20location_department%2C%20%20location_region%2C%20location_countrycode%2C%20%20country_fr&where=location_city%20%3D%20%22"+location_name+"%22%20AND%20lastdate_end%3E%3D%22"+date_passee+"T00%3A00%3A00%2B00%3A00%22&limit=-1&lang="+lang+"&timezone=UTC&use_labels=false&compressed=false&epsg=4326"
    try:
        response = requests.get(url)
        df = pd.DataFrame(response.json())
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None
        
    return df

def extract_gps_coordinates(df:pd.DataFrame,column="location_coordinates"):
    try:
        df['latitude'] = df[column].apply(lambda x: x['lat']).astype('Float64')
        df['longitude'] = df[column].apply(lambda x: x['lon']).astype('Float64')
        df = df.drop(column, axis=1)
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        print("Vérifiez les données")
        return None
    return df

def coordinates_in_frontiere(df:pd.DataFrame,frontiere):
    """
    Sert à garder les coordonnées gps qui sont dans un polygone donné (frontière)
    """
    try:
        
        valid_mask = ((df['latitude'].notna())|(df['longitude'].notna())) # On vérifie que les coordonnées GPS ne sont pas nulles
        rejected_coordinates = df.loc[~valid_mask].copy()
        
        candidates = df.loc[valid_mask].copy()

        # On crée des objets Point à partir des coordonnées GPS
        points = candidates.apply(
        lambda row: Point(row["longitude"], row["latitude"]),
        axis=1,
        )
        # On vérifie si les points sont dans la frontière
        in_frontiere_mask = points.map(frontiere.covers) # covers() est utilisé pour inclure les points sur la frontière elle-même. Pour exclure ces points, utilisez contains() à la place.

        events_in_frontiere = candidates.loc[in_frontiere_mask].copy()
        events_outside_frontiere = candidates.loc[~in_frontiere_mask].copy()
 

    except Exception as e:
        print(f"Erreur de l'exécution de la fonction : {e}")
        print("Vérifier si les colonnes latitude et longitude existent bien \n Vérifier la nature de la frontière.")
    
    return events_in_frontiere, rejected_coordinates, events_outside_frontiere

# Dictionnaire couple nom et les code postal et code administratif de chaque arrondissement de Paris
CODES_POSTAUX_PARIS = {}
CODES_POSTAUX_PARIS["paris 1er arrondissement"] = ["75001","75101"]
for i in range(2, 21):
    CODES_POSTAUX_PARIS[f"paris {i}e arrondissement"] = [f"750{i:02d}",f"751{i:02d}"]
        
def coordinates_in_arr(df:pd.DataFrame,df_frontiere_arr:pd.DataFrame,carte, column_location:str, column_polygon:str, column_code:str, column_arrondissement:str,code_beginning="75", codes_ref=CODES_POSTAUX_PARIS):
    """
    Sert à mapper les événements dans les bons arrondissements en fonction de leur code postal
    
    Paramètres
    ----------
    df : DataFrame
        Événements contenant latitude, longitude et code postal.
    df_frontiere_arr : DataFrame ou GeoDataFrame
        Frontières géographiques des arrondissements.
    carte:
        Carte sur laquelle afficher les points.
    column_location : str
        Colonne contenant le nom du lieu
    column_polygon : str
        Colonne contenant les polygones Shapely.
    column_code : str
        Colonne contenant le code postal des événements.
    column_arrondissement : str
        Colonne contenant le nom de chaque arrondissement.
    code_beginning : str
        Début du code postal
    codes_ref : dict
        Dictionnaire associant chaque arrondissement à ses codes postaux.


    """
    try:
        resultats = df.copy()
       # Colonnes pour stocker les résultats
        resultats["arrondissement_gps"] = None
        resultats["code_postal_correct"] = False
        resultats["ancien_code_postal"] = None

        for idx, row in resultats.iterrows():
            # On crée des objets Point à partir des coordonnées GPS de chaque événements
            point = Point(row["longitude"], row["latitude"])
            
            # On cherche l'arrondissement qui contient le point
            for _, arr in df_frontiere_arr.iterrows():
                if arr[column_polygon].covers(point):
    
                    nom_arr = arr[column_arrondissement]
                    
                    # Enregistrer l'arrondissement trouvé
                    resultats.at[idx, "arrondissement_gps"] = nom_arr

                    # Lire le code postal depuis le dataframe de résultats
                    code_postal = resultats.at[idx, column_code]

                    # Si le code postal est manquant, le remplacer par la 1re valeur du dictionnaire
                    if pd.isna(code_postal):
                        resultats.at[idx, "ancien_code_postal"] = "manquant"
                        code_postal = codes_ref[nom_arr][0]
                        resultats.at[idx, column_code] = code_postal
                    # Si le code postal ne correspond pas à la position mais commence par 75, remplacer par la 1re valeur du dictionnaire
                    if code_postal is not None and code_postal.startswith(code_beginning):
                        if code_postal not in codes_ref[nom_arr]:
                            resultats.at[idx, "ancien_code_postal"] = code_postal
                            code_postal = codes_ref[nom_arr][0]
                            resultats.at[idx, column_code] = code_postal
                    # Vérifier si le code postal correspond
                    resultats.at[idx, "code_postal_correct"] = (code_postal in codes_ref[nom_arr])

                    break

        resultats_correct = resultats.loc[resultats["code_postal_correct"]]
        resultats_incorrect = resultats.loc[~resultats["code_postal_correct"]]
        
        for _, row in resultats_correct.iterrows():
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=5,
                popup=(
                    f"Lieu : {row[column_location]}<br>"
                    f"Arrondissement GPS : {row['arrondissement_gps']}<br>"
                    f"Code postal correct : {row['code_postal_correct']}"
                ),
                color="green",
                fill=True,
            ).add_to(carte)

        for _, row in resultats_incorrect.iterrows():
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=5,
                popup=(
                    f"Lieu : {row[column_location]}<br>"
                    f"Arrondissement GPS : {row['arrondissement_gps']}<br>"
                    f"Code postal correct : {row['code_postal_correct']}<br>"
                    f"Code postal: {row[column_code]}"
                ),
                color="red",
                fill=True,
            ).add_to(carte)
            
    except Exception as e:        
        print(f"Erreur de l'exécution de la fonction : {e}")
        raise
    
    
    return resultats, resultats_correct, resultats_incorrect, carte