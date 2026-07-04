# 🤖 Adaptive RAG System con LangGraph, OpenAI & Tavily

Sistema de Recuperación Aumentada Generativa (RAG) **adaptativo** implementado con LangGraph, OpenAI y Tavily Search.

El agente evalúa cada pregunta y decide dinámicamente si recuperar información de documentos locales (vectorstore) o buscar en la web en tiempo real, aplicando además verificación de alucinaciones antes de responder.

---

## 🧠 Contexto del Curso — RAG Tradicional vs. Adaptive RAG vs. RAG Vectorless

En clase estudiamos tres enfoques de RAG con diferentes casos de uso:

| Técnica | Embeddings | Cuándo usar |
|---|---|---|
| **RAG Tradicional** | ✅ Sí | Documentos no estructurados, preguntas directas |
| **Adaptive RAG** ← *este proyecto* | ✅ Sí | Cuando necesitas decidir entre múltiples fuentes |
| **RAG Vectorless / Page Index** | ❌ No | Documentos jerárquicos (papers, reportes financieros, manuales) |

### ¿Por qué RAG Vectorless es relevante?
El Page Index (técnica vista en clase) alcanza un **98% de precisión** en benchmarks financieros vs. 50% del RAG tradicional, preservando la jerarquía del documento (sección → subsección → anexos) sin necesidad de embeddings ni vectorstore.

> **Este proyecto implementa Adaptive RAG**, que es el enfoque pedido en la tarea — pero conocer las tres técnicas es clave para elegir la correcta según el caso de negocio.

---

## 📐 Arquitectura del Agente

```
Usuario
   │
   ▼
[Router] ─────────────────────────────────┐
   │ (pregunta sobre docs locales)        │ (pregunta general / web)
   ▼                                      ▼
[Vectorstore FAISS]              [Tavily Web Search]
   │                                      │
   ▼                                      │
[Grader de Relevancia] ◄──────────────────┘
   │
   ├── Relevante ──────► [Generator] ──► [Hallucination Check] ──► Respuesta
   │
   └── No relevante ──► [Tavily Web Search] ──► [Generator] ──► Respuesta
```

### Flujo del grafo LangGraph — 6 nodos

| Nodo | Función |
|---|---|
| **Route Question** | Clasifica la pregunta: vectorstore o búsqueda web |
| **Retrieve** | Recupera documentos del vectorstore (FAISS) |
| **Grade Documents** | Evalúa si los documentos son relevantes para la pregunta |
| **Web Search** | Fallback con Tavily si los docs no son relevantes |
| **Generate** | Genera la respuesta con el contexto recuperado |
| **Hallucination Check** | Verifica que la respuesta esté basada en hechos reales |

---

## 📁 Estructura del Proyecto (dentro del monorepo GenAI-main)

```
GenAI-main/
├── pyproject.toml            # Dependencias de TODO el monorepo (Poetry)
└── Agent/Langgraph/Adaptive_rag/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── agents/
    │   │   ├── __init__.py
    │   │   └── adaptive_rag.py       # Grafo principal LangGraph
    │   ├── chains/
    │   │   ├── __init__.py
    │   │   ├── router.py             # Clasificador de preguntas
    │   │   ├── grader.py             # Evaluadores de relevancia y alucinaciones
    │   │   └── generator.py          # Chain de generación RAG
    │   ├── retriever/
    │   │   ├── __init__.py
    │   │   └── vectorstore.py        # Configuración FAISS + carga de documentos
    │   └── utils/
    │       ├── __init__.py
    │       └── helpers.py            # Tavily Search tool
    │
    ├── notebooks/
    │   └── adaptive_rag_legacy_notebook.ipynb   # Notebook original (referencia)
    │
    ├── tests/
    │   ├── __init__.py
    │   └── test_rag.py               # Tests básicos del grafo
    │
    ├── graph.jpeg / graph2.jpeg / graph3.jpeg    # Diagramas del grafo
    ├── .env.example                  # Variables de entorno de ejemplo
    └── README.md
```

---

## 🖥️ Backend + Frontend (Chat UI)

Además del agente como librería, el proyecto incluye una app web completa:

```
Adaptive_rag/
├── backend/
│   ├── main.py         # API FastAPI (sesiones, chats, mensajes)
│   ├── database.py     # Persistencia SQLite de chats permanentes
│   └── chats.db         # (se genera en runtime, ignorado por git)
└── frontend/
    └── index.html       # Chat UI (HTML/CSS/JS vanilla, sin build tools)
```

**Cómo funciona:**
- **Chat temporal** (⚡): sesión en memoria del servidor. Se pierde al reiniciar el backend.
- **Chat permanente** (🔒): cada mensaje se guarda en SQLite (`backend/chats.db`); aparece en la lista de la izquierda y sobrevive a reinicios.

### Levantar la app

```bash
cd Agent/Langgraph/Adaptive_rag/backend
poetry run uvicorn main:app --reload --port 8000
```

Abre **http://localhost:8000** — el mismo servidor FastAPI sirve la API y el frontend (mismo origen, sin problemas de CORS).

### Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/sessions/temporary` | Crea sesión temporal |
| POST | `/api/chats` | Crea chat permanente |
| GET | `/api/chats` | Lista chats permanentes |
| GET | `/api/chats/{id}/messages` | Historial de un chat |
| DELETE | `/api/chats/{id}` | Elimina un chat permanente |
| POST | `/api/message` | Envía mensaje al agente `{session_id, message, mode}` |

Docs interactivos automáticos de FastAPI: **http://localhost:8000/docs**

---



Este proyecto vive dentro del monorepo `GenAI-main`, en `Agent/Langgraph/Adaptive_rag/`, y usa las dependencias gestionadas por Poetry en la raíz del repo (no tiene su propio venv ni requirements.txt).

### 1. Instalar dependencias (desde la raíz del monorepo)
```bash
cd GenAI-main
poetry install
```

### 2. Configurar variables de entorno
```bash
cd Agent/Langgraph/Adaptive_rag
cp .env.example .env
# Edita .env con tus API keys (OPENAI_API_KEY, TAVILY_API_KEY)
```

### 3. Ejecutar (desde la raíz del monorepo, para que los imports `src.*` resuelvan)
```bash
poetry run python -c "
from Agent.Langgraph.Adaptive_rag.src.agents.adaptive_rag import AdaptiveRAGAgent
agent = AdaptiveRAGAgent()
print(agent.invoke('¿Qué es LangGraph y cómo funciona?'))
"
```

O añade `Agent/Langgraph/Adaptive_rag` a tu `PYTHONPATH` y usa directamente:
```python
from src.agents.adaptive_rag import AdaptiveRAGAgent
```

---

## ⚙️ Variables de Entorno

```env
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here
```

- OpenAI API Key: https://platform.openai.com/api-keys
- Tavily API Key: https://app.tavily.com

---

## 🧰 Tecnologías

| Tecnología | Uso |
|---|---|
| **LangGraph** | Orquestación del grafo de agentes con estado |
| **LangChain** | Chains, prompts y herramientas |
| **OpenAI GPT-4o-mini** | Modelo de lenguaje (router, grader, generator) |
| **Tavily Search** | Búsqueda web en tiempo real |
| **FAISS** | Vectorstore local para recuperación semántica |
| **Python 3.10+** | Lenguaje base |

---

## 💡 Lecciones del Curso aplicadas en este proyecto

### 1. Empezar por lo más fácil y determinístico
El router clasifica primero las preguntas simples (vectorstore) antes de ir a búsqueda web, reduciendo alucinaciones.

### 2. Medir precisión antes de escalar
El `retrieval_grader` evalúa cada documento antes de usarlo, y el `hallucination_grader` verifica la respuesta final — siguiendo el principio de no confiar ciegamente en el modelo.

### 3. MVP primero
El proyecto cubre el caso base (preguntas sobre LangChain/LangGraph) con documentos locales. Escalar a más fuentes es el siguiente paso natural.

### 4. Evitar alucinaciones
El grafo tiene un nodo dedicado a verificar que la respuesta esté fundamentada en los documentos recuperados, no inventada por el LLM.

### 5. Diseño modular
Cada componente (router, grader, generator, retriever) está separado en su propio módulo, facilitando pruebas y reemplazo individual.

---

## 📊 Ejemplo de decisión del agente

```
Pregunta: "¿Qué es LangGraph?"
→ Router: vectorstore (tema en los docs locales)
→ Retrieve: 4 documentos de langchain-ai.github.io
→ Grade: 3 documentos relevantes
→ Generate: respuesta basada en los 3 docs
→ Hallucination check: ✓ fundamentada
→ Answer check: ✓ responde la pregunta
→ Respuesta final al usuario ✅

Pregunta: "¿Cuál es la tasa de inflación en Perú hoy?"
→ Router: web_search (dato en tiempo real)
→ Tavily: busca en la web
→ Generate: respuesta con fuentes actuales
→ Hallucination check: ✓ fundamentada en resultados web
→ Respuesta final al usuario ✅
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📖 Referencias

- [Artículo base — Building an Adaptive RAG System](https://levelup.gitconnected.com/building-an-adaptive-rag-system-with-langgraph-openai-and-tavily-c4ee39d2f021)
- [Documentación LangGraph](https://langchain-ai.github.io/langgraph/)
- [RAG Vectorless / Page Index](https://medium.com/) — técnica estudiada en clase
- [Tavily Search API](https://tavily.com)

---

## 👤 Autor

**Jeancarlo**
GitHub: [@JeancarloJGCO](https://github.com/JeancarloJGCO)
