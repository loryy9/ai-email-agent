import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Config
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DB_PATH = os.getenv("DB_PATH", "./chroma_db_pro")
RISPOSTE_DIR = "./data/risposte"
FEEDBACK_FILE = "./data/feedback/errate.jsonl"
DB_CFG = {"hnsw:space": "cosine"}

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model=os.getenv("LLM_MODEL", "llama3-70b-8192"), temperature=0)

def salva_risposta_corretta(domanda, risposta):
    # 1. Backup su file
    os.makedirs(RISPOSTE_DIR, exist_ok=True)
    filename = f"risposta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(os.path.join(RISPOSTE_DIR, filename), "w", encoding="utf-8") as f:
        f.write(f"DOMANDA: {domanda}\nRISPOSTA: {risposta}")

    # 2. Iniezione Live nel DB
    db = Chroma(collection_name="storico_risposte", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=DB_CFG)
    db.add_documents([Document(page_content=f"D: {domanda}\nR: {risposta}", metadata={"fonte": filename})])
    print(f"✨ [LIVE MEMORY] Risposta salvata e imparata.")

def salva_feedback_errato(domanda, risposta_agente, motivazione, corretta=None):
    # 1. Backup su JSONL
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), "domanda": domanda, "risposta_agente": risposta_agente, "motivazione": motivazione, "risposta_corretta": corretta}
    
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Iniezione Live nel DB Errori
    db = Chroma(collection_name="feedback_errori", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=DB_CFG)
    db.add_documents([Document(page_content=f"DOMANDA: {domanda}\nERRORE: {risposta_agente}\nMOTIVO: {motivazione}", metadata={"tipo": "errore"})])
    print(f"🚫 [LIVE MEMORY] Errore registrato. Non lo ripeterò.")

def cerca_contesto(query):
    # Ricerca incrociata (Storico + Documentazione)
    db_risp = Chroma(collection_name="storico_risposte", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=DB_CFG)
    db_doc = Chroma(collection_name="documentazione", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=DB_CFG)
    
    # Prendiamo k=10 dalla doc per non perdere le voci nelle tabelle
    res_risp = db_risp.similarity_search_with_relevance_scores(query, k=2)
    res_doc = db_doc.similarity_search_with_relevance_scores(query, k=10)

    # Debug in console
    if res_doc: print(f"🔍 [DEBUG] Top Doc Score: {res_doc[0][1]:.4f} ({res_doc[0][0].metadata.get('source')})")

    storico = [d[0].page_content for d in res_risp if d[1] > 0.80]
    manuali = [d[0].page_content for d in res_doc if d[1] > 0.20]

    contesto = ""
    if storico: contesto += "--- RISPOSTE CORRETTE PASSATE ---\n" + "\n".join(storico) + "\n\n"
    if manuali: contesto += "--- ESTRATTI MANUALI TECNICI ---\n" + "\n".join(manuali)
    
    return contesto

def recupera_errori(query):
    db = Chroma(collection_name="feedback_errori", embedding_function=embeddings, persist_directory=DB_PATH, collection_metadata=DB_CFG)
    res = db.similarity_search(query, k=2)
    return "\n\n".join([d.page_content for d in res]) if res else "Nessun errore simile."

def esegui_agente(mail):
    contesto = cerca_contesto(mail)
    errori = recupera_errori(mail)

    prompt = f"""Sei l'assistente tecnico senior di CL SYSTEM. Rispondi alla MAIL usando il CONTESTO.

### REGOLE OBBLIGATORIE:
1. Usa **TABELLE MARKDOWN** per codici voce o tracciati record.
2. Sii schematico e usa il **grassetto** per i codici (es. **0880**).
3. Se il CONTESTO non contiene la soluzione, rispondi: **ESCALATE**.
4. NON ripetere gli errori elencati in 'ERRORI DA EVITARE'.

### ERRORI DA EVITARE:
{errori}

### CONTESTO:
{contesto if contesto else "VUOTO - Rispondi ESCALATE"}

### MAIL CLIENTE:
{mail}

RISPOSTA PROFESSIONALE:"""

    return llm.invoke(prompt).content

if __name__ == "__main__":
    mail_test = input("\n📩 Mail cliente: ")
    print("... Studio in corso ...")
    
    bozza = esegui_agente(mail_test)
    print(f"\n--- PROPOSTA ---\n\n{bozza}\n\n{'-'*40}")
    
    f = input("\n❓ Corretta (c) o Errata (e)? ").lower().strip()
    if f == 'c':
        salva_risposta_corretta(mail_test, bozza)
    elif f == 'e':
        m = input("✍️ Motivo errore: ")
        c = input("✅ Risposta corretta (opzionale): ")
        salva_feedback_errato(mail_test, bozza, m, c if c else None)
        if c: salva_risposta_corretta(mail_test, c)