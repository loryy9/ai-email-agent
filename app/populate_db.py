import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "./chroma_db_local"
FEEDBACK_FILE = "./data/feedback/errate.jsonl"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def importa_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print(f"Feedback file non trovato: {FEEDBACK_FILE}")
        return

    db_err = Chroma(
        collection_name="feedback_errori",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    db_risp = Chroma(
        collection_name="storico_risposte",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    docs_err = []
    docs_ok = []

    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"Riga {i} non valida, salto.")
                continue

            domanda = r.get("domanda", "")
            risposta_agente = r.get("risposta_agente", "")
            motivazione = r.get("motivazione", "")
            risposta_corretta = (r.get("risposta_corretta") or "").strip()

            testo_err = (
                f"DOMANDA: {domanda}\n"
                f"RISPOSTA_SBAGLIATA: {risposta_agente}\n"
                f"MOTIVO_ERRORE: {motivazione}\n"
                f"RISPOSTA_CORRETTA: {risposta_corretta or 'N/D'}"
            )
            docs_err.append(
                Document(
                    page_content=testo_err,
                    metadata={"tipo": "errore", "source": "errate.jsonl", "row": i}
                )
            )

            if risposta_corretta:
                docs_ok.append(
                    Document(
                        page_content=f"D: {domanda}\nR: {risposta_corretta}",
                        metadata={"fonte": f"feedback_row_{i}"}
                    )
                )

    if docs_err:
        db_err.add_documents(docs_err)
    if docs_ok:
        db_risp.add_documents(docs_ok)

    print(f"Importati feedback errori: {len(docs_err)}")
    print(f"Importate risposte corrette da feedback: {len(docs_ok)}")

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
    importa_feedback()