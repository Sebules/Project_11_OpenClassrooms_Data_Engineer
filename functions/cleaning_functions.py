
import re
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata


def remove_duplicates(df:pd.DataFrame,subset)->pd.DataFrame:
    """
    Permet de supprimer les lignes en doublon d'un dataframe donné pour un subset donné.
    """
    try:
        print(f"Il y a {df.duplicated(subset).sum()} lignes en doublon dans le dataframe")
        print(f"\n dimensions avant suppression:{df.shape}")
        df.drop_duplicates(subset=subset,ignore_index=True)
        print(f"\n dimensions après suppression:{df.shape}")
    except Exception as e:
        print(f"Something goes wrong: {e}")
    return df

def delete_nulls(df:pd.DataFrame,column)->pd.DataFrame:
    """
    Permet de supprimer les lignes avec des données nulles dans une colonne donnée.
    On récupère le dataframe propre et le dataframe avec les valeurs nulles pour vérification si besoin.
    """
    try:
        print(f"Il y a {df[column].isna().sum()} valeurs nulles dans la colonne {column}")
        print(f"\n dimensions avant suppression:{df.shape}")
        df_clean = df[~df[column].isna()].reset_index(drop=True)
        df_nulls = df[df[column].isna()].reset_index(drop=True)
        print(f"\n dimensions après suppression:{df_clean.shape}")
    except Exception as e:
        print(f"Something goes wrong: {e}")
    return df_clean, df_nulls

def complete_nan(df:pd.DataFrame,column,value)->pd.DataFrame:
    """
    Sert à compléter les NaN d'une colonne avec des valeurs
    """
    try:
        print(f"Il y a {df[column].isna().sum()} valeurs nulles dans la colonne {column}")
        df[column] = df[column].fillna(value)
        print(f"Il y a maintenant {df[column].isna().sum()} valeurs nulles dans la colonne {column}")
    except Exception as e:
        print(f"Something goes wrong: {e}")
    return df

def beautifulsoup_clean(x):
    """
    Extrait le texte HTML et normalise les espaces.
    """
    # Ne traiter que les chaînes de caractères
    if not isinstance(x, str):
        return x

    # Créer un objet BeautifulSoup pour analyser le HTML
    soup = BeautifulSoup(x,'html.parser')
    
    # Supprimer les contenus inutiles
    for tag in soup(["script", "style", "noscript"]): # Supprimer les balises script, style et noscript
        tag.decompose() # Supprimer le contenu de la balise
    
    # Extraire le texte en séparant les éléments HTML
    x = soup.get_text(separator=" ", strip=True)
    
    # Normaliser certains caractères
    x = x.replace("\xa0"," ") # Remplacer les espaces insécables par des espaces normaux
    x = x.replace("\u2028","\n") # Ligne séparateur
    x = x.replace("\u2029", "\n") # Paragraphe séparateur
    
    # Supprimer les caractères Unicode de formatage
    x = "".join(
        char for char in x
        if unicodedata.category(char) != "Cf" # Cf = Other, Format. unicodedata.category(char) renvoie la catégorie Unicode d'un caractère. "Cf" correspond aux caractères de formatage, qui sont généralement invisibles et n'ont pas de représentation graphique. En filtrant ces caractères, on obtient un texte plus propre et plus lisible.
    )

    # Remplacer les tabulations et espaces multiples
    x = re.sub(r"[^\S\n]+", " ", x) # Remplacer les espaces multiples par un seul espace, mais conserver les retours à la ligne

    # Réduire les retours à la ligne multiples
    x = re.sub(r"\n+", "\n", x) # Remplacer les retours à la ligne multiples par un seul retour à la ligne

    return x.strip() # Strip pour supprimer les espaces en début et fin de chaîne
    

def remove_balises_html(df:pd.DataFrame,column)->pd.DataFrame:
    """
    Sert à retirer les balises HTML et autres symboles d'un texte dans un dataframe
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas.")

    df[column] = df[column].apply(lambda x : beautifulsoup_clean(x))
    return df

def convert_to_datetime(df:pd.DataFrame, columns)->pd.DataFrame:
    """
    Sert à la conversation en date de colonnes
    """
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df