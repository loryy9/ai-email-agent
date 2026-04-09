import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Setup costanti
DB_PATH = "./chroma_db_local"
RISPOSTE_DIR = "./data/risposte"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def salva_risposta_corretta(domanda, risposta):
    """Salva la risposta approvata nel DB e su file .txt"""
    # 1. Salva su file .txt per permanenza
    if not os.path.exists(RISPOSTE_DIR):
        os.makedirs(RISPOSTE_DIR)
    
    filename = f"risposta_{len(os.listdir(RISPOSTE_DIR)) + 1}.txt"
    with open(os.path.join(RISPOSTE_DIR, filename), "w", encoding="utf-8") as f:
        f.write(f"DOMANDA: {domanda}\nRISPOSTA: {risposta}")

    # 2. Aggiungi al Database vettoriale (collezione storico_risposte)
    db_risp = Chroma(collection_name="storico_risposte", embedding_function=embeddings, persist_directory=DB_PATH)
    db_risp.add_documents([Document(page_content=f"D: {domanda} R: {risposta}", metadata={"fonte": filename})])
    print(f"\n✅ Memoria aggiornata! Creata fonte: {filename}")

def cerca_contesto(query):
    # Cerca prima nello storico (priorità alta)
    db_risp = Chroma(collection_name="storico_risposte", embedding_function=embeddings, persist_directory=DB_PATH)
    risultati = db_risp.similarity_search_with_relevance_scores(query, k=1)
    
    if risultati and risultati[0][1] > 0.8: # Soglia alta per lo storico
        return risultati[0][0].page_content, "STORICO RISPOSTE"

    # Poi cerca nella documentazione
    db_doc = Chroma(collection_name="documentazione", embedding_function=embeddings, persist_directory=DB_PATH)
    risultati_doc = db_doc.similarity_search(query, k=3)
    contesto = "\n---\n".join([d.page_content for d in risultati_doc])
    return contesto, "DOCUMENTAZIONE TECNICA"

def esegui_agente(mail_cliente):
    contesto, fonte = cerca_contesto(mail_cliente)
    
    prompt = f"""
    Sei un assistente tecnico. Rispondi alla mail basandoti solo sul CONTESTO fornito.
    Se il contesto non è pertinente, scrivi 'ESCALATE'.
    
    CONTESTO ({fonte}): {contesto}
    MAIL CLIENTE: {mail_cliente}
    RISPOSTA PROFESSIONALE:
    """
    
    risposta = llm.invoke(prompt).content
    return risposta, fonte

if __name__ == "__main__":
    mail_test = input("\n📧 Inserisci il testo della mail ricevuto: ")
    
    print("\n... L'agente sta studiando la risposta ...")
    bozza, fonte = esegui_agente(mail_test)
    
    print(f"\n--- PROPOSTA DELL'AGENTE (Fonte: {fonte}) ---")
    print(bozza)
    print("-" * 40)
    
    feedback = input("\n❓ Questa risposta è 'corretta' o 'errata'? ").strip().lower()
    
    if feedback == "corretta":
        salva_risposta_corretta(mail_test, bozza)
        print("Ottimo! La prossima volta saprò già cosa rispondere.")
    else:
        print("❌ Operazione annullata. La risposta non è stata salvata.")