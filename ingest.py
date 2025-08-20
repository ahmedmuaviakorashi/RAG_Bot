import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

CSV_PATH = r"D:\Confiz\Project 3- AI Bot O'LLAMA\rag\docs\data.csv"      
INDEX_DIR = "faiss_index"
EMBED_MODEL = "mxbai-embed-large" 

def main():
    if not os.path.isfile(CSV_PATH):
        raise RuntimeError(f"CSV not found at {CSV_PATH}")

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Converting CSV rows to text...")
    rows_as_text = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        rows_as_text.append(text)

    print(f"Total rows: {len(rows_as_text)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.create_documents(rows_as_text)

    print("Embedding and building FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)

    print(f"Saving index to {INDEX_DIR}/")
    vs.save_local(INDEX_DIR)

    print("Done. Data Vectorized")

if __name__ == "__main__":
    main()

