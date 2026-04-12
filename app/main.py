import os
import re
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
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DOC_K = int(os.getenv("DOC_K", "6"))
DOC_SCORE_MIN = float(os.getenv("DOC_SCORE_MIN", "0.35"))
STORICO_SCORE_MIN = float(os.getenv("STORICO_SCORE_MIN", "0.72"))
STORICO_DOC_MARGIN = float(os.getenv("STORICO_DOC_MARGIN", "0.03"))
ERROR_K = int(os.getenv("ERROR_K", "3"))
ERROR_SCORE_MIN = float(os.getenv("ERROR_SCORE_MIN", "0.45"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


EMBED_NORMALIZE = _env_bool("EMBED_NORMALIZE", True)
ENABLE_MODEL_ROUTING = _env_bool("ENABLE_MODEL_ROUTING", True)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE}
)

llm_quality = ChatGroq(model=LLM_MODEL, temperature=0)
llm_fast = ChatGroq(model=TRIAGE_MODEL, temperature=0)


def _append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _next_risposta_filename() -> str:
    max_id = 0
    for nome in os.listdir(RISPOSTE_DIR):
        match = re.fullmatch(r"risposta_(\d+)\.txt", nome)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"risposta_{max_id + 1}.txt"


def _format_docs(docs_con_score, prefisso: str) -> str:
    blocchi = []
    for idx, (doc, score) in enumerate(docs_con_score, start=1):
        fonte = doc.metadata.get("fonte") or doc.metadata.get("source") or "sconosciuta"
        blocchi.append(
            f"[{prefisso} {idx} | score={score:.2f} | fonte={fonte}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocchi)


def salva_risposta_corretta(domanda, risposta):
    """Salva la risposta approvata nel DB e su file .txt"""
    if not os.path.exists(RISPOSTE_DIR):
        os.makedirs(RISPOSTE_DIR)

    filename = _next_risposta_filename()
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
    # 1) Cerca nella documentazione principale
    db_doc = Chroma(
        collection_name="documentazione",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    risultati_doc = db_doc.similarity_search_with_relevance_scores(query, k=DOC_K)
    doc_validi = [(doc, score) for doc, score in risultati_doc if score >= DOC_SCORE_MIN]
    best_doc_score = doc_validi[0][1] if doc_validi else 0.0

    # 2) Cerca nello storico risposte corrette
    db_risp = Chroma(
        collection_name="storico_risposte",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    risultati_storico = db_risp.similarity_search_with_relevance_scores(query, k=2)
    storico_top = risultati_storico[0] if risultati_storico else None
    storico_score = storico_top[1] if storico_top is not None else 0.0
    storico_affidabile = storico_score >= STORICO_SCORE_MIN

    if doc_validi:
        contesto = _format_docs(doc_validi, "DOC")
        # Aggiungiamo lo storico solo quando ha score vicino/migliore della doc.
        if storico_top is not None and storico_affidabile and storico_score >= (best_doc_score - STORICO_DOC_MARGIN):
            contesto += "\n\n---\n\n" + _format_docs([storico_top], "STORICO")
        return contesto, "DOCUMENTAZIONE TECNICA"

    if storico_top is not None and storico_affidabile:
        return _format_docs([storico_top], "STORICO"), "STORICO RISPOSTE"

    return "", "NESSUNA FONTE AFFIDABILE"


def recupera_errori_simili(query, k=ERROR_K):
    db_err = Chroma(
        collection_name="feedback_errori",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    risultati = db_err.similarity_search_with_relevance_scores(query, k=k)
    validi = [(doc, score) for doc, score in risultati if score >= ERROR_SCORE_MIN]
    if not validi:
        return "Nessun errore simile registrato."
    return _format_docs(validi, "ERRORE_SIMILE")


def _scegli_llm(mail_cliente: str, fonte: str, errori_simili: str):
    if not ENABLE_MODEL_ROUTING:
        return llm_quality, LLM_MODEL

    if fonte != "STORICO RISPOSTE":
        return llm_quality, LLM_MODEL

    parole = len(mail_cliente.split())
    richiesta_semplice = parole <= 35 and mail_cliente.count("?") <= 1 and "\n" not in mail_cliente
    no_errori_noti = errori_simili == "Nessun errore simile registrato."

    if richiesta_semplice and no_errori_noti:
        return llm_fast, TRIAGE_MODEL

    return llm_quality, LLM_MODEL


def _estrai_testo_llm(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parti = []
        for item in content:
            if isinstance(item, str):
                parti.append(item)
            elif isinstance(item, dict) and "text" in item:
                parti.append(str(item.get("text", "")))
        return "\n".join(p for p in parti if p).strip()

    return str(content).strip()


def esegui_agente(mail_cliente):
    contesto, fonte = cerca_contesto(mail_cliente)
    errori_simili = recupera_errori_simili(mail_cliente, k=ERROR_K)
    llm_attivo, modello_attivo = _scegli_llm(mail_cliente, fonte, errori_simili)

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
5) Se NON rispondi ESCALATE, chiudi con: "Fonte usata: <STORICO RISPOSTE|DOCUMENTAZIONE TECNICA>"
6) Se rispondi ESCALATE, non aggiungere altro testo.

ERRORI DA EVITARE:
{errori_simili}

CONTESTO ({fonte}):
{contesto}

MAIL CLIENTE:
{mail_cliente}

RISPOSTA PROFESSIONALE:
"""
    output = llm_attivo.invoke(prompt)
    risposta = _estrai_testo_llm(output.content)

    # Normalizza l'output in caso di risposta di escalation verbosa.
    if risposta.upper().startswith("ESCALATE"):
        risposta = "ESCALATE"

    return risposta, fonte, modello_attivo


if __name__ == "__main__":
    mail_test = input("\n📧 Inserisci il testo della mail ricevuto: ")

    print("\n... L'agente sta studiando la risposta ...")
    bozza, fonte, modello = esegui_agente(mail_test)

    print(f"\n--- PROPOSTA DELL'AGENTE (Fonte: {fonte} | Modello: {modello}) ---")
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
