import os
import time
import requests
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 

# --- CONFIGURATION ---
DATA_PATH = "Data" 
DB_PATH = "legal_chroma_db"
MODEL_NAME = "all-minilm"

# --- CUSTOM CLASS TO FORCE CONNECTION TO PORT 11434 ---
class ManualOllama:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = "http://127.0.0.1:11434/api/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Force raw HTTP request to the correct port
            response = requests.post(
                self.base_url,
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                results.append(response.json()["embedding"])
            else:
                print(f"Error embedding chunk: {response.text}")
                # Return empty list or zero vector on failure to prevent crash
                results.append([]) 
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def create_vector_db():
    print(f"1. Loading PDF files from '{DATA_PATH}'...")
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"   - Loaded {len(documents)} pages.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   - Created {len(chunks)} text chunks.")

    print("3. Creating embeddings (Using Manual Bypass)...")
    
    # USE THE MANUAL CLASS
    embeddings = ManualOllama(model_name=MODEL_NAME)
    
    # Initialize DB
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )

    # --- BATCH PROCESSING ---
    batch_size = 10
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"   - Processing batch {i//batch_size + 1}/{(total_chunks//batch_size)+1} ({len(batch)} chunks)...")
        
        try:
            vector_store.add_documents(documents=batch)
            # No sleep needed for raw requests usually, but let's keep a tiny one
            time.sleep(0.1) 
        except Exception as e:
            print(f"   ! Error on batch starting at index {i}: {e}")
            break

    print(f"   - Success! Database saved to '{DB_PATH}'.")

if __name__ == "__main__":
    create_vector_db()