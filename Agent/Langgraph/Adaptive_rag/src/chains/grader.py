

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ─────────────────────────────
# RETRIEVAL GRADER
# ─────────────────────────────

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Debe responder únicamente 'yes' o 'no'."
    )


retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un Retrieval Grader de un sistema RAG.

Tu única tarea es decidir si el documento contiene información útil para responder la pregunta.

Reglas:

- Responde únicamente "yes" o "no".
- Responde "yes" si el documento contiene información relacionada, aunque sea parcial.
- Responde "yes" si menciona directamente el tema preguntado.
- Responde "no" solamente cuando el documento sea completamente irrelevante.

No respondas la pregunta.
No expliques tu decisión.
"""
    ),
    (
        "human",
        """
Pregunta:
{question}

Documento:
{document}
"""
    ),
])

retrieval_grader = retrieval_prompt | llm.with_structured_output(GradeDocuments)

# GRADER

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Debe responder únicamente 'yes' o 'no'."
    )


hallucination_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un Hallucination Grader.

Debes verificar si la respuesta está sustentada por los documentos.

Responde:

- "yes" si la respuesta puede justificarse usando los documentos.
- "no" si contiene información inventada o no respaldada.

No expliques.
"""
    ),
    (
        "human",
        """
Documentos:
{documents}

Respuesta:
{generation}
"""
    ),
])

hallucination_grader = (
    hallucination_prompt
    | llm.with_structured_output(GradeHallucinations)
)

# ANSWER

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Debe responder únicamente 'yes' o 'no'."
    )


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres un Answer Grader.

Debes decidir si la respuesta responde la pregunta del usuario.

Responde:

- "yes" si responde correctamente.
- "no" si no responde o es irrelevante.

No expliques.
"""
    ),
    (
        "human",
        """
Pregunta:
{question}

Respuesta:
{generation}
"""
    ),
])

answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)