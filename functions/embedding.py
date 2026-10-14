from sentence_transformers import SentenceTransformer
import pandas as pd

def embedding_st(df:pd.DataFrame,column:str, model:str):

    """
    Sert à faire de l'embedding d'un texte dans un dataframe avec SentenceTransformer

    Paramètres
    ----------
    df : DataFrame
        Dataframe qui contient les segments à transformer en vecteur
    column : string
        colonne qui contient les segments
    model : string
        Modèle LLM utilisé
    """

    # Charger le modèle d'embedding
    modele = SentenceTransformer(model)
    # Vectoriser le texte en une seule opération
    listes = df[column].tolist()
    textes = [seg for ligne in listes for seg in ligne]          # aplatir
    vecteurs = modele.encode(textes, show_progress_bar=True)     # un seul appel
    resultats, i = [], 0
    for ligne in listes:                                         # redécouper par ligne
        resultats.append(vecteurs[i:i + len(ligne)].tolist())
        i += len(ligne)
    df["embedding"] = resultats
    # Aperçu du résultat avec les vecteurs
    print(df[[column, "embedding"]].iloc[0])

    return df["embedding"], df