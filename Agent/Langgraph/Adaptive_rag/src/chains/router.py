

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class RouteQuery(BaseModel):
    datasource: str = Field(description="web_search o vectorstore")


router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Clasifica la pregunta:

vectorstore → IA, LangGraph, LangChain, RAG, embeddings
web_search → actualidad o temas generales

Responde solo: vectorstore o web_search
"""),
    ("human", "{question}"),
])

question_router = router_prompt | llm.with_structured_output(RouteQuery)