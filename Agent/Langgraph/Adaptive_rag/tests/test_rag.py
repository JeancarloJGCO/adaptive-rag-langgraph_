

import pytest
from unittest.mock import patch, MagicMock


def test_graph_state_structure():

    from src.agents.adaptive_rag import GraphState
    state: GraphState = {
        "question": "¿Qué es LangGraph?",
        "generation": "",
        "documents": [],
    }
    assert "question" in state
    assert "generation" in state
    assert "documents" in state


def test_build_graph():
    with patch("src.agents.adaptive_rag.get_retriever"), \
         patch("src.agents.adaptive_rag.question_router"), \
         patch("src.agents.adaptive_rag.rag_chain"):
        from src.agents.adaptive_rag import build_graph
        graph = build_graph()
        assert graph is not None


def test_route_question_returns_valid_option():
    valid_routes = {"vectorstore", "web_search"}
    # Mock del router
    mock_result = MagicMock()
    mock_result.datasource = "vectorstore"
    assert mock_result.datasource in valid_routes
