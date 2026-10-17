import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.extraction_opendata import retrieve_opendata,extract_gps_coordinates, coordinates_in_frontiere, coordinates_in_arr
from functions.polygon_paris import retrieve_polygon_paris, retrieve_polygon_arr_paris
from functions.cleaning_functions import remove_duplicates, delete_nulls, remove_balises_html,complete_nan
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from functions.chunking import chunk_rcts
from functions.embedding import embedding_st
from functions.indexation import indexation_faiss

MODE_DEV = True      # passer à False pour l'exécution complète
TAILLE_ECHANTILLON = 100

if __name__ == "__main__":
    print("\n=====Import des données=====\n")
    df= retrieve_opendata()
    print("Données importées depuis Opendata")

    print("\n=====Séparation des latitude et longitude=====\n")
    df_gps= extract_gps_coordinates(df)
    print("longitude et latitude séparées")

    print("\n=====Suppression des événements géographiquement hors de Paris=====\n")
    # Récupération du polygone de Paris
    carte,polygon_paris,shape_polygon = retrieve_polygon_paris()
    print("Polygone Paris récupéré")

    events_in_frontiere, rejected_coordinates, events_outside_frontiere=coordinates_in_frontiere(df_gps,shape_polygon)
    print("Uniquement les événements à Paris ont été gardés")

    print("\n=====Nettoyage=====\n")
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

    print("\n=====Sauvegarde du dataframe en .csv====\n")
    df_clean6.to_csv(f'dataset_opendatasoft_evenements_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv',header=True, encoding='utf-8', index=False)
    print(f'fichier dataset_opendatasoft_evenements_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv sauvegardé.')

    print("\n=====Vectorisation=====\n")
    df_v = df_clean6.copy()

    if MODE_DEV:
        df_v = df_v.sample(n=min(TAILLE_ECHANTILLON, len(df_v)), random_state=42)
        print(f"MODE DEV : échantillon de {len(df_v)} lignes")
        df_v = df_v.reset_index(drop=True)

    print("\n____Combinaisondu texte_____\n")
    df_v["texte_combine"] = (
    "Titre : " + df_v["title_fr"]
    + " - Description : " + df_v["longdescription_fr"]
    + " - Conditions : " + df_v["conditions_fr"]
    + " - Mots clés : " + df_v["keywords_fr"]
    + " - Lieu : " + df_v["location_name"]
    + " - Adresse : " + df_v["location_address"]
    + " - Arrondissement : " + df_v["arrondissement_gps"]
    + " - Code postal : " + df_v["location_postalcode"]
    + " - Ville : " + df_v ["location_city"]
     )

    print(f"Combinaison de texte réalisée. Exemple avec la première ligne du dataframe:\n {df_v["texte_combine"][0]}")

    print("\n____Chunking_____\n")
    segments,df_c = chunk_rcts(1000,200,df_v,column="texte_combine")

    print("\n____Embedding_____\n")
    embedding,df_e=embedding_st(df=df_c,column="segments", model="paraphrase-multilingual-MiniLM-L12-v2")

    print("\n=====Indexation=====\n")
    df_e["latitude"] = pd.to_numeric(df_e["latitude"], errors="coerce")
    df_e["longitude"] = pd.to_numeric(df_e["longitude"], errors="coerce")
    df_e["lastdate_begin"] = pd.to_datetime(df_e["lastdate_begin"], errors="coerce", utc=True)
    df_e["lastdate_end"] = pd.to_datetime(df_e["lastdate_end"], errors="coerce", utc=True)
    metadata_plus = ["latitude","longitude","lastdate_begin","lastdate_end"]
    
    textes,vecteurs, metadatas = indexation_faiss(
        df=df_e,
        segments_column="segments",
        embedding_column="embedding",
        metadata_columns = metadata_plus)
    print("Premier vecteur : ",vecteurs[0])
    print("Premier metadonnees : ",metadatas[0])

    

    
    