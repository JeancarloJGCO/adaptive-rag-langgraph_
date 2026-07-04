
from pathlib import Path

import requests
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv

load_dotenv()

# Fuentes: Markdown crudo y estático (no requiere ejecutar JS para verlo)
URLS = [
    "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
]

# Dónde persistir el índice FAISS (evita reconstruir embeddings en cada reinicio)
INDEX_DIR = Path(__file__).parent / "faiss_index"

HEADERS_TO_SPLIT_ON = [
    ("#", "titulo_h1"),
    ("##", "titulo_h2"),
    ("###", "titulo_h3"),
]

_retriever = None


def _fetch_markdown(url: str) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": "adaptive-rag-bot"}, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  No se pudo descargar {url}: {e}")
        return None


def _load_and_split_documents() -> list[Document]:
    print("Cargando documentos...")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    splits: list[Document] = []
    for url in URLS:
        markdown = _fetch_markdown(url)
        if markdown is None:
            continue

        header_chunks = header_splitter.split_text(markdown)

        for chunk in header_chunks:
            chunk.metadata["source"] = url
            splits.extend(size_splitter.split_documents([chunk]))

    if not splits:
        raise RuntimeError(
            "No se pudo cargar ningún documento para el vectorstore. "
            "Revisa tu conexión a internet y que las URLs en vectorstore.py sean accesibles."
        )

    return splits


def build_vectorstore() -> FAISS:
    splits = _load_and_split_documents()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"  Vectorstore creado con {len(splits)} chunks")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"  Índice persistido en {INDEX_DIR}")

    return vectorstore


def load_or_build_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings()
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        print(f"Cargando vectorstore persistido desde {INDEX_DIR}...")
        try:
            return FAISS.load_local(
                str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"  No se pudo cargar el índice persistido ({e}), reconstruyendo...")
    return build_vectorstore()


def get_retriever(k: int = 4):
    global _retriever
    if _retriever is None:
        vs = load_or_build_vectorstore()

        faiss_retriever = vs.as_retriever(search_kwargs={"k": k})

        all_docs = list(vs.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k

        _retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
    return _retriever
