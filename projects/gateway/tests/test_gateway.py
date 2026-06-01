import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from langchain_core.messages import HumanMessage

# import our application artifacts
from src.main import create_app
from src.services.llm_service import agent
from src.core.config import settings

# ---------------------------------------------------------
# 1. LANGGRAPH ENGINE INTEGRATION TESTS
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_langgraph_agent_state_mutation():
    """
    Ensures that compiling and invoking the graph accepts a HumanMessage
    and appends a valid AIMessage to the state track dictionary.
    """

    # Setup state input payload
    initial_input = {
        "messages": [HumanMessage(content="What is 5 multiplied by 5?")],
        "llm_calls": 0
    }

    # Run the compiled agent machine topology
    final_state = await agent.ainvoke(initial_input)
    
    # Structural Assertions
    assert "messages" in final_state
    assert len(final_state["messages"]) >= 2
    
    # Extract terminal response message
    final_message = final_state["messages"][-1]
    
    # Verify properties match structural contracts
    assert final_message.content != ""
    assert isinstance(final_message.content, str)
    
    # Verify model evaluation updated state attributes if tracked
    if "llm_calls" in final_state:
        assert final_state["llm_calls"] >= 0


# ---------------------------------------------------------
# 2. FASTAPI ENDPOINT REST E2E TESTS
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_api_chat_endpoint_success():
    """
    E2E simulation hitting the router over async HTTP 
    to assert full schema compliance.
    """
    app = create_app()
    
    # Use AsyncClient with an ASGI transport to test the app without spinning up a live network port
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"prompt": "Tell me a 1-sentence joke."}
        
        response = await ac.post("/api/v1/chat", json=payload)
        
        # Assert structural HTTP rules
        assert response.status_code == 200
        
        json_data = response.json()
        
        # Assert schema keys conform completely
        assert "response" in json_data
        assert "provider" in json_data
        assert isinstance(json_data["response"], str)
        
        # Assert provider matches current environment settings
        assert json_data["provider"] == settings.LLM_PROVIDER.lower()


@pytest.mark.asyncio
async def test_api_chat_endpoint_validation_failure():
    """
    Asserts validation layer catches malformed payload bodies cleanly.
    """
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Sending an empty object should cause a Pydantic parsing failure (422 Unprocessable Entity)
        response = await ac.post("/api/v1/chat", json={})
        assert response.status_code == 422