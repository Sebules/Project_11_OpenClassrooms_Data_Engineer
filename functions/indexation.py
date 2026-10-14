import faiss
import pandas as pd
import numpy as np

  
def indexation_faiss(df:pd.DataFrame,segments_column:str, embedding_column:str):
    textes, vecteurs, metadatas = [], [], []
    for i, row in df.iterrows():
        for j, (seg, vec) in enumerate(zip(row[segment_column], row[embedding_column])):
            textes.append(seg)                     # seg est un str
            vecteurs.append(vec)
            metadatas.append({"doc_id": i, "chunk_id": j})   # ajout ici d'autres colonnes utiles
            
    vecteurs = np.array(vecteurs, dtype="float32")

    
    print(len(textes), vecteurs.shape," : comparaison taille des textes et taille des vecteurs")

    print("\n__Index faiss__\n")
    faiss.normalize_L2(vecteurs)   # cosinus = produit scalaire
    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)
    print(index.ntotal," : nombre d'index")

    return index, vecteurs

    
    