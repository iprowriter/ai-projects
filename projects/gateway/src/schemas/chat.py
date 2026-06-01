from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(
    ..., 
    min_length=4, 
    max_length=100, 
    description="The user prompt to process", 
    json_schema_extra={"example": "What is capital of Germany"}
)

class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated output text string from the LLM provider.")
    provider: str = Field(..., description="Identifies which provider handled the request ('gemini' or 'ollama').")