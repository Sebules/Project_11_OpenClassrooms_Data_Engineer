import pandas as pd
from functions.extraction_opendata import retrieve_opendata,extract_gps_coordinates, coordinates_in_frontiere, coordinates_in_arr
from functions.polygon_paris import retrieve_polygon_paris, retrieve_polygon_arr_paris
from functions.cleaning_functions import remove_duplicates, delete_nulls, remove_balises_html,complete_nan
from datetime import datetime

if __name__ == "__main__":
    print("=====Import des données=====")
    df= retrieve_opendata()
    print("Données importées depuis Opendata")

    print("=====Séparation des latitude et longitude=====")
    df_gps= extract_gps_coordinates(df)
    print("longitude et latitude séparées")

    print("=====Suppression des événements géographiquement hors de Paris=====")
    # Récupération du polygone de Paris
    carte,polygon_paris,shape_polygon = retrieve_polygon_paris()
    print("Polygone Paris récupéré")

    events_in_frontiere, rejected_coordinates, events_outside_frontiere=coordinates_in_frontiere(df_gps,shape_polygon)
    print("Uniquement les événements à Paris ont été gardés")

    print("=====Nettoyage=====")
    df_clean = remove_duplicates(df=events_in_frontiere,subset="uid")
    print("Il n'y a pas de doublons")

    
    df_clean2,df_nulls = delete_nulls(df=df_clean,column="description_fr")
    print("Les lignes avec une description vide ont été supprimées")

    df_clean3 = complete_nan(df_clean2, column="longdescription_fr",value=df_clean2["description_fr"])
    print("Les descriptions longues vides ont été complétées")

    df_clean4 = remove_balises_html(df_clean3,"longdescription_fr")
    print("Les balises HTML ont été supprimées")
    
    carte,df_paris = retrieve_polygon_arr_paris()
    print("Les polygones des arrondissements de Paris sont chargés")

    resultats, resultats_correct, resultats_incorrect, carte = coordinates_in_arr(
    df_clean4, df_paris, carte, "location_name","geometry","location_postalcode", "com_arm_name_lower")
    print("Récupération des événements qui ont des coordonnées gps et code postal corrigés et corrects")

    df_clean5 = resultats_correct.copy()

    df_clean5.loc[df_clean5['conditions_fr'].isna()==True,'conditions_fr']="Pas de condition"
    print("Conditions avec NaN complétées")
    
    df_clean6 = df_clean5.copy()
    df_clean6["keywords_fr"] = df_clean6["keywords_fr"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_clean6["keywords_fr"]=df_clean6["keywords_fr"].apply(lambda x: " ".join(x) if isinstance(x, list) else " ")
    print("Les mots-clés ont été convertis en chaîne de caractères")

    print("=====Sauvegarde du dataframe en .csv====")
    df_clean6.to_csv(f'dataset_opendatasoft_evenements_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',header=True, encoding='utf-8', index=False)
    print(f'fichier dataset_opendatasoft_evenements_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv sauvegardé.')