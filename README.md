# GenAI — Adaptive RAG

Sistema RAG adaptativo con **LangGraph**, **OpenAI** y **Tavily Search**, con backend (FastAPI) y frontend (Chat UI) incluidos.

El agente decide dinámicamente si responder desde un vectorstore local o buscar en la web, evalúa sus propias respuestas (alucinación + relevancia) y reintenta antes de responder — con corte automático de reintentos para evitar loops infinitos.

---

## Requisitos previos

- **Python 3.12 o 3.13**
- **[Poetry](https://python-poetry.org/docs/#installation)** (gestor de dependencias)
- Cuenta y API key de **[OpenAI](https://platform.openai.com/api-keys)**
- Cuenta y API key de **[Tavily](https://app.tavily.com/)** (tiene tier gratuito)

Verifica que tienes Poetry instalado:
```bash
poetry --version
```
Si no lo tienes:
```bash
pip install poetry
```

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd GenAI-main
```

### 2. Instalar dependencias
```bash
poetry lock
poetry install
```

### 3. Configurar las API keys

**Windows (CMD):**
```cmd
cd Agent\Langgraph\Adaptive_rag
copy .env.example .env
notepad .env
```

**Windows (PowerShell) / Linux / macOS:**
```bash
cd Agent/Langgraph/Adaptive_rag
cp .env.example .env    # PowerShell: Copy-Item .env.example .env
```

Edita `.env` y coloca tus claves reales:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
```

> El archivo `.env` está en `.gitignore` — nunca se sube al repo. Cada persona que clone el proyecto debe crear el suyo con sus propias claves.

---

## Levantar el proyecto

```bash
cd Agent/Langgraph/Adaptive_rag/backend
poetry run uvicorn main:app --reload --port 8000
```

Abre en el navegador:

```
http://localhost:8000
```

Ahí verás la Chat UI completa (sidebar con chats temporales/permanentes). Un solo proceso sirve la API y el frontend — no hace falta levantar nada más.

**Nota sobre el primer mensaje:** la primera vez que envíes una pregunta, el backend descarga y vectoriza los documentos fuente (LangGraph/LangChain docs) y guarda el índice en `src/retriever/faiss_index/`. Tarda más esa vez; los siguientes reinicios son instantáneos porque el índice queda persistido en disco.

Docs interactivos de la API (Swagger): `http://localhost:8000/docs`

---

## Estructura del proyecto

```
GenAI-main/
├── pyproject.toml                    # Dependencias (Poetry)
└── Agent/Langgraph/Adaptive_rag/
    ├── .env.example                  # Plantilla de variables de entorno
    ├── src/
    │   ├── agents/adaptive_rag.py    # Grafo principal LangGraph (nodos + routing)
    │   ├── chains/
    │   │   ├── router.py             # Clasifica: vectorstore vs web_search
    │   │   ├── grader.py             # Evalúa relevancia / alucinación / utilidad
    │   │   └── generator.py          # Genera la respuesta (con citación de fuentes)
    │   ├── retriever/vectorstore.py  # FAISS + BM25 (hybrid search) + chunking jerárquico
    │   └── utils/helpers.py          # Tavily search tool
    ├── backend/
    │   ├── main.py                   # API FastAPI
    │   └── database.py               # Persistencia SQLite de chats permanentes
    ├── frontend/index.html           # Chat UI (HTML/CSS/JS, sin build tools)
    └── tests/test_rag.py             # Tests con pytest
```

---

## Correr los tests

```bash
poetry run pytest Agent/Langgraph/Adaptive_rag/tests/ -v
```