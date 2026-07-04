from typing import List, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START

from ..chains.router import question_router
from ..chains.grader import retrieval_grader, hallucination_grader, answer_grader
from ..chains.generator import rag_chain, format_docs_with_sources
from ..retriever.vectorstore import get_retriever
from ..utils.helpers import web_search_tool



# Estado del grafo

class GraphState(TypedDict):
    question:   str
    generation: str
    documents:  List[Document]
    retries:    int


MAX_RETRIES = 3



def retrieve(state: GraphState) -> GraphState:
    print("RETRIEVE")

    question = state["question"]

    retriever = get_retriever()
    documents = retriever.invoke(question)

    print(f"Se recuperaron {len(documents)} documentos")

    for i, doc in enumerate(documents, start=1):
        print(f"\n----- Documento {i} -----")
        print(doc.page_content[:500])
        print("-------------------------")

    return {
        "documents": documents,
        "question": question,
    }


def generate(state: GraphState) -> GraphState:
    print("GENERATE")
    question  = state["question"]
    documents = state["documents"]
    retries   = state.get("retries", 0) + 1

    context = format_docs_with_sources(documents)
    generation = rag_chain.invoke({"context": context, "question": question})

    return {"documents": documents, "question": question, "generation": generation, "retries": retries}


def grade_documents(state: GraphState) -> GraphState:
    print("GRADE DOCUMENTS")
    question  = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score == "yes":
            print(f"  Documento relevante")
            filtered_docs.append(doc)
        else:
            print(f"  Documento no relevante")

    return {"documents": filtered_docs, "question": question}


def web_search(state: GraphState) -> GraphState:
    print("WEB SEARCH")

    question = state["question"]
    documents = state.get("documents", [])

    results = web_search_tool.invoke({"query": question})

    if isinstance(results, list):
        page_content = "\n\n".join(
            f"Título: {item.get('title', '')}\n"
            f"Contenido: {item.get('content', '')}\n"
            f"URL: {item.get('url', '')}"
            for item in results
            if isinstance(item, dict)
        )
    else:
        page_content = str(results)

    documents.append(Document(page_content=page_content))

    return {
        "documents": documents,
        "question": question,
    }

# Condicionales

def route_question(state: GraphState) -> Literal["vectorstore", "web_search"]:
    print("ROUTE QUESTION")
    question = state["question"]
    source   = question_router.invoke({"question": question})

    if source.datasource == "web_search":
        print("  Ruta: WEB SEARCH")
        return "web_search"
    else:
        print("  Ruta: VECTORSTORE")
        return "vectorstore"


def decide_to_generate(state: GraphState) -> Literal["generate", "web_search"]:
    print("DECIDE TO GENERATE")
    documents = state["documents"]

    if not documents:
        print("  Sin documentos relevantes")
        return "web_search"
    else:
        print("  Documentos relevantes encontrados")
        return "generate"


def grade_generation(state: GraphState) -> Literal["useful", "not useful", "not supported"]:
    print("GRADE GENERATION")
    question   = state["question"]
    documents  = state["documents"]
    generation = state["generation"]
    retries    = state.get("retries", 0)

    if retries >= MAX_RETRIES:
        print(f"  Límite de {MAX_RETRIES} reintentos alcanzado")
        return "useful"

    # Verificar alucinaciones
    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_score.binary_score == "yes":
        # Verificar si responde la pregunta
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_score.binary_score == "yes":
            print("  Respuesta fundamentada")
            return "useful"
        else:
            print("  Respuesta no responde la pregunta")
            return "not useful"
    else:
        print("  Alucinación")
        return "not supported"


# Grafo

def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    # Agregar nodos
    workflow.add_node("web_search",      web_search)
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate",        generate)

    # Punto de entrada con routing condicional
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search":  "web_search",
            "vectorstore": "retrieve",
        }
    )

    # Flujo principal
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve",   "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate":   "generate",
        }
    )

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "not useful":    "web_search",
            "useful":        END,
        }
    )

    return workflow.compile()

# Clase prnicipal

class AdaptiveRAGAgent:

    def __init__(self):
        self.app = build_graph()

    def invoke(self, question: str) -> str:
        inputs = {"question": question, "retries": 0}
        result = self.app.invoke(inputs, config={"recursion_limit": 50})
        return result.get("generation", "No se pudo generar una respuesta.")

    def stream(self, question: str):
        inputs = {"question": question, "retries": 0}
        for output in self.app.stream(inputs, config={"recursion_limit": 50}):
            for key, value in output.items():
                print(f"\n--- Nodo: {key} ---")
            print("\n---\n")


if __name__ == "__main__":
    agent = AdaptiveRAGAgent()
    response = agent.invoke("¿Qué es LangGraph?")
    print(f"\nRespuesta: {response}")
