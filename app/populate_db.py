import os
import json
import glob
import shutil
import hashlib
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Config
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DB_PATH = os.getenv("DB_PATH", "./chroma_db_pro")
DOCS_DIR = "./data/doc"
FEEDBACK_FILE = "./data/feedback/errate.jsonl"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def _load_text_files(base_dir: str) -> List[Document]:
    files = glob.glob(os.path.join(base_dir, "**/*.txt"), recursive=True)
    docs = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    docs.append(Document(page_content=content, metadata={"source": os.path.basename(fp)}))
                    print(f"📄 Caricato: {fp}")
        except Exception as e:
            print(f"❌ Errore caricamento {fp}: {e}")
    return docs

def _load_feedback(path: str):
    err_docs, ok_docs = [], []
    if not os.path.exists(path): return err_docs, ok_docs
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                txt_err = f"DOMANDA: {r['domanda']}\nERRORE: {r['risposta_agente']}\nCORREZIONE: {r.get('motivazione')}"
                err_docs.append(Document(page_content=txt_err, metadata={"tipo": "errore"}))
                if r.get("risposta_corretta"):
                    ok_docs.append(Document(page_content=f"D: {r['domanda']}\nR: {r['risposta_corretta']}", metadata={"tipo": "storico"}))
            except:
                continue
    return err_docs, ok_docs

def main():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🧹 Pulizia: Vecchio database rimosso.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    cfg = {"hnsw:space": "cosine"}

    db_doc = Chroma(collection_name="documentazione", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=cfg)
    db_err = Chroma(collection_name="feedback_errori", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=cfg)
    db_risp = Chroma(collection_name="storico_risposte", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=cfg)

    # 1. Manuali
    raw_docs = _load_text_files(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(raw_docs)
    if chunks:
        db_doc.add_documents(chunks)
        print(f"✅ Documentazione: {len(chunks)} chunk inseriti.")

    # 2. Feedback precedenti
    err_docs, ok_docs = _load_feedback(FEEDBACK_FILE)
    if err_docs: db_err.add_documents(err_docs)
    if ok_docs: db_risp.add_documents(ok_docs)
    print(f"✅ Feedback: {len(err_docs)} errori e {len(ok_docs)} successi importati.")

    print(f"\n🚀 [DONE] DB pronto! (Doc: {db_doc._collection.count()} | Storico: {db_risp._collection.count()})")

if __name__ == "__main__":
    main()