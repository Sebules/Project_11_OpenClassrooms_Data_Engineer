import faiss
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
  
def indexation_faiss(df:pd.DataFrame,segments_column:str, embedding_column:str,metadata_columns: list[str] | None = None):
    """
    Crée un index FAISS à partir des embeddings, et conserve en parallèle
    les textes et les métadonnées de chaque chunk (même ordre que l'index).

    Paramètres
    ----------
    df : DataFrame
        Contient les segments, les embeddings et les colonnes de métadonnées
    segments_column : str
        Colonne contenant les listes de segments
    embedding_column : str
        Colonne contenant les listes de vecteurs
    metadata_columns : list[str]
        Colonnes à stocker en métadonnées (par exemple: dates, coordonnées, etc.)
    
    Retourne:
    ----------
    textes,
    index,
    vecteurs,
    metadatas
    """
    try:
        if metadata_columns is None:
            metadata_columns = []
            
        textes, vecteurs, metadatas = [], [], []
        for i, row in df.iterrows():
            # Métadonnées communes à tous les chunks d'un même événement
            meta_evenement = {col: row[col] for col in metadata_columns}
            for j, (seg, vec) in enumerate(zip(row[segments_column], row[embedding_column])):
                textes.append(seg)                     # seg est un str
                vecteurs.append(vec)
                metadatas.append({"doc_id": i, "chunk_id": j}| meta_evenement)   # ajout ici d'autres colonnes utiles
                
        vecteurs = np.array(vecteurs, dtype="float32")
        
        print(len(textes), vecteurs.shape," : comparaison taille des textes et taille des vecteurs")
        if len(textes) == vecteurs.shape[0]:
            print("La vectorisation s'est bien passée, le nombre de segments correspond au nombre de vecteurs")
        else:
            print("La vectorisation ne s'est pas bien passée, le nombre de segments ne correspond pas au nombre de vecteurs")
            
        print("\n__Index faiss__\n")
        faiss.normalize_L2(vecteurs)   # cosinus = produit scalaire
        index = faiss.IndexFlatIP(vecteurs.shape[1])
        index.add(vecteurs)
        print(index.ntotal," : nombre total d'index")

    except Exception as e:
        print(f"Something goes wrong: {e}")
        
    return textes, index, vecteurs, metadatas

def indexation_lc_faiss(df:pd.DataFrame,segments_column:str, embedding_column:str,embeddings,metadata_columns: list[str] | None = None):
    """
    Crée un index FAISS  avec LangChain à partir des embeddings, et conserve en parallèle
    les textes et les métadonnées de chaque chunk (même ordre que l'index).

    Paramètres
    ----------
    df : DataFrame
        Contient les segments, les embeddings et les colonnes de métadonnées
    segments_column : str
        Colonne contenant les listes de segments
    embedding_column : str
        Colonne contenant les listes de vecteurs
    embeddings : objet LangChain Embeddings (ex. HuggingFaceEmbeddings)
        utilise UNIQUEMENT pour vectoriser les questions a la recherche. Il doit etre le meme modele que celui
        des embeddings de la colonne, avec encode_kwargs={"normalize_embeddings": True}.
    metadata_columns : list[str]
        Colonnes à stocker en métadonnées (par exemple: dates, coordonnées, etc.)
    
    Retourne:
    ----------
    textes,
    vectorstore,
    vecteurs,
    metadatas
    """
    try:
        if metadata_columns is None:
            metadata_columns = []
            
        textes, vecteurs, metadatas = [], [], []
        for i, row in df.iterrows():
            # Métadonnées communes à tous les chunks d'un même événement
            meta_evenement = {col: row[col] for col in metadata_columns}
            for j, (seg, vec) in enumerate(zip(row[segments_column], row[embedding_column])):
                textes.append(seg)                     # seg est un str
                vecteurs.append(vec)
                metadatas.append({"doc_id": i, "chunk_id": j}| meta_evenement)   # ajout ici d'autres colonnes utiles
                
        vecteurs = np.array(vecteurs, dtype="float32")
        
        
        print(len(textes), vecteurs.shape," : comparaison taille des textes et taille des vecteurs")
        if len(textes) == vecteurs.shape[0]:
            print("La vectorisation s'est bien passée, le nombre de segments correspond au nombre de vecteurs")
        else:
            print("La vectorisation ne s'est pas bien passée, le nombre de segments ne correspond pas au nombre de vecteurs")

        print("\n__Index faiss__\n")
        faiss.normalize_L2(vecteurs) 
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(textes, vecteurs.tolist())),
            embedding=embeddings,        # votre wrapper SentenceTransformer, pour les questions
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        print(vectorstore.index.ntotal," : nombre total d'index")
        
        nb_docs = len({m["doc_id"] for m in metadatas})
        print(f"{nb_docs} evenements indexes sur {len(df)} dans le DataFrame")
    except Exception as e:
        print(f"Something goes wrong: {e}")
        
    return textes, vectorstore, vecteurs, metadatas    