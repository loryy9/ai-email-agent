import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Setup costanti
DB_PATH = "./chroma_db_local"
RISPOSTE_DIR = "./data/risposte"
FEEDBACK_DIR = "./data/feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "errate.jsonl")

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DOC_K = int(os.getenv("DOC_K", "6"))
DOC_SCORE_MIN = float(os.getenv("DOC_SCORE_MIN", "0.35"))

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)
llm = ChatGroq(model=LLM_MODEL, temperature=0)


def _append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def salva_risposta_corretta(domanda, risposta):
    """Salva la risposta approvata nel DB e su file .txt"""
    if not os.path.exists(RISPOSTE_DIR):
        os.makedirs(RISPOSTE_DIR)

    filename = f"risposta_{len(os.listdir(RISPOSTE_DIR)) + 1}.txt"
    with open(os.path.join(RISPOSTE_DIR, filename), "w", encoding="utf-8") as f:
        f.write(f"DOMANDA: {domanda}\nRISPOSTA: {risposta}")

    db_risp = Chroma(
        collection_name="storico_risposte",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    db_risp.add_documents([
        Document(
            page_content=f"D: {domanda}\nR: {risposta}",
            metadata={"fonte": filename}
        )
    ])
    print(f"\n✅ Memoria aggiornata! Creata fonte: {filename}")


def salva_feedback_errato(domanda, risposta_agente, motivazione, risposta_corretta=None):
    """Salva feedback negativo + motivazione su file e su DB vettoriale."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "domanda": domanda,
        "risposta_agente": risposta_agente,
        "motivazione": motivazione,
        "risposta_corretta": risposta_corretta or ""
    }
    _append_jsonl(FEEDBACK_FILE, record)

    db_err = Chroma(
        collection_name="feedback_errori",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    testo = (
        f"DOMANDA: {domanda}\n"
        f"RISPOSTA_SBAGLIATA: {risposta_agente}\n"
        f"MOTIVO_ERRORE: {motivazione}\n"
        f"RISPOSTA_CORRETTA: {risposta_corretta or 'N/D'}"
    )
    db_err.add_documents([
        Document(page_content=testo, metadata={"tipo": "errore"})
    ])
    print("📝 Feedback errato salvato (file + memoria vettoriale).")


def cerca_contesto(query):
    # 1) Cerca nello storico risposte corrette (priorità alta)
    db_risp = Chroma(
        collection_name="storico_risposte",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    risultati = db_risp.similarity_search_with_relevance_scores(query, k=2)
    if risultati and risultati[0][1] > 0.70:
        return risultati[0][0].page_content, "STORICO RISPOSTE"

    # 2) Cerca nella documentazione
    db_doc = Chroma(
        collection_name="documentazione",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    risultati_doc = db_doc.similarity_search_with_relevance_scores(query, k=DOC_K)
    validi = [doc for doc, score in risultati_doc if score >= DOC_SCORE_MIN]
    contesto = "\n---\n".join([d.page_content for d in validi]) if validi else ""
    return contesto, "DOCUMENTAZIONE TECNICA"


def recupera_errori_simili(query, k=2):
    db_err = Chroma(
        collection_name="feedback_errori",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    docs = db_err.similarity_search(query, k=k)
    if not docs:
        return "Nessun errore simile registrato."
    return "\n\n".join([d.page_content for d in docs])


def esegui_agente(mail_cliente):
    contesto, fonte = cerca_contesto(mail_cliente)
    errori_simili = recupera_errori_simili(mail_cliente, k=2)

    prompt = f"""
Sei un assistente tecnico email.
Rispondi SOLO usando il CONTESTO. Non inventare.

Regole obbligatorie:
1) Se il contesto è vuoto o insufficiente, rispondi esattamente: ESCALATE
2) Usa prima il CONTESTO e poi, solo se coerente, lo STORICO.
3) Evita gli errori nel blocco ERRORI DA EVITARE.
4) Se dai una procedura:
codice menu - descrizione menu
- passo 1
- passo 2
- passo 3
5) Chiudi con: "Fonte usata: <STORICO RISPOSTE|DOCUMENTAZIONE TECNICA>"

ERRORI DA EVITARE:
{errori_simili}

CONTESTO ({fonte}):
{contesto}

MAIL CLIENTE:
{mail_cliente}

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
        motivo = input("✍️ Perché è errata? (obbligatorio): ").strip()
        corretta = input("✅ Se vuoi, incolla qui una risposta corretta (opzionale): ").strip()

        salva_feedback_errato(
            domanda=mail_test,
            risposta_agente=bozza,
            motivazione=motivo if motivo else "Motivazione non fornita",
            risposta_corretta=corretta if corretta else None
        )

        # Se l'operatore fornisce la risposta corretta, la salviamo anche come memoria positiva
        if corretta:
            salva_risposta_corretta(mail_test, corretta)
            print("✅ Salvata anche la risposta corretta suggerita dall'operatore.")
        else:
            print("❌ Risposta non salvata come corretta (mancava testo corretto).")