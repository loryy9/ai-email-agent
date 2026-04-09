import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Setup: Modello di embedding locale (gratis e gira sulla tua CPU)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_PATH = "./chroma_db_local"

def carica_dati(cartella, nome_collezione):
    if not os.path.exists(cartella):
        print(f"⚠️ Cartella {cartella} non trovata. Creala e mettici dei file .txt")
        os.makedirs(cartella)
        return

    # Leggiamo i file .txt
    documenti = []
    for nome_file in os.listdir(cartella):
        if nome_file.endswith(".txt"):
            percorso = os.path.join(cartella, nome_file)
            with open(percorso, "r", encoding="utf-8") as f:
                testo = f.read()
                documenti.append(Document(page_content=testo, metadata={"fonte": nome_file}))

    if not documenti:
        print(f"❌ Nessun file .txt trovato in {cartella}")
        return

    # Dividiamo il testo in piccoli pezzi (Chunking)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    pezzi = splitter.split_documents(documenti)

    # Salviamo nel Database locale
    Chroma.from_documents(
        documents=pezzi,
        embedding=embeddings,
        collection_name=nome_collezione,
        persist_directory=DB_PATH
    )
    print(f"✅ Caricati {len(pezzi)} pezzi nella collezione '{nome_collezione}'")

if __name__ == "__main__":
    print("Inizio popolamento database...")
    carica_dati("./data/doc", "documentazione")
    carica_dati("./data/risposte", "storico_risposte")