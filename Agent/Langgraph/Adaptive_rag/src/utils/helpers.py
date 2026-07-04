

try:
    # Paquete recomendado (langchain >= 0.3.25)
    from langchain_tavily import TavilySearch as TavilySearchResults
    _TAVILY_KWARGS = {"max_results": 3, "search_depth": "advanced"}
except ImportError:
    # Fallback para versiones antiguas de langchain-community
    from langchain_community.tools.tavily_search import TavilySearchResults
    _TAVILY_KWARGS = {"max_results": 3, "search_depth": "advanced"}

from dotenv import load_dotenv

load_dotenv()

# búsqueda web
web_search_tool = TavilySearchResults(**_TAVILY_KWARGS)
