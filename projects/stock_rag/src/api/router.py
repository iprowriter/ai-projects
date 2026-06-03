from fastapi import APIRouter, HTTPException
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.llm_service import generate_llm_response

api_router = APIRouter(prefix="/api/v1")


@api_router.post("/chat", response_model=ChatResponse)
async def handle_chat_request(payload: ChatRequest):
    try:
        # Hand off payload to your freshly compiled LangGraph service loop!
        agent_result = await generate_llm_response(payload.prompt)

        return ChatResponse(
            response=agent_result["text"],
            provider=agent_result["provider"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Agent Loop Failure: {str(e)}")