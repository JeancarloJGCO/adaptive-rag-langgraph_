from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Responde usando SOLO el contexto proporcionado.

Si no está la respuesta, di:
"No encuentro esa información en el contexto."

Sé claro y breve.

Al final de tu respuesta, agrega una línea "Fuentes:" listando el/los
"source" (URL) de los fragmentos de contexto que realmente usaste para
responder. Si no usaste ninguno (porque no encontraste la respuesta),
omite esa línea.
"""),
    ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
])

rag_chain = rag_prompt | llm | StrOutputParser()


def format_docs_with_sources(docs) -> str:

    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "desconocida")
        titulo = doc.metadata.get("titulo_h2") or doc.metadata.get("titulo_h1") or ""
        header = f" — sección: {titulo}" if titulo else ""
        blocks.append(f"[source: {source}{header}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)