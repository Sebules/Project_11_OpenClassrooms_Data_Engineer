import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def print_segments(segments):
    for i, segment in enumerate(segments):
        print(f"Segment {i+1}:\n{segment}\n")
        
def chunk_rcts(chunk_size:int,chunk_overlap:int, df:pd.DataFrame,column:str):
    """
    Sert à réaliser des chunks sur un dataframe avec RecursiveCharacterTextSplitter().

    Paramètres
    ----------
    chunk_size :  integer
        Taille de chaque segment
    chunk_overlap: integer
        Chevauchement entre les segments
    df : DataFrame
        Dataframe qui contient les textes à chunker
    column : string
        colonne qui contient les textes

    Retourne:
    ----------
    df['segments']
    df
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Taille de chaque segment
        chunk_overlap=chunk_overlap # Chevauchement entre les segments
        )
    df['segments'] = df[column].apply(lambda x: text_splitter.split_text(x))

    print("Résultat du chunking pour la première ligne du dataframe\n")
    print_segments(df['segments'][0])

    return df['segments'], df