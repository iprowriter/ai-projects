from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from src.core.config import settings



# Dynamic AI Model Selector
def _init_configured_model():
    """Dynamically resolves the model provider from environment configurations."""
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        return init_chat_model(
            settings.GEMINI_MODEL,
            model_provider="google_genai",
            temperature=0,
            api_key=settings.GOOGLE_API_KEY,
        )
    elif provider == "ollama":
        return init_chat_model(
            settings.OLLAMA_MODEL, # Uses "gemma3:270m"
            model_provider="ollama",
            temperature=0,
            # base_url=settings.OLLAMA_BASE_URL -> init_chat_model automatically defaults to http://localhost:11434
        )
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")

# 1. Initialize the unified model with System Instructions pre-bound
# This guarantees your system instructions are never duplicated in the state list!
raw_model = _init_configured_model()

# Define operations tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers a and b."""
    return a + b

tools = [multiply, add]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to the initialized base client
model_with_tools = raw_model.bind_tools(tools)

# 2. Define Clean State Layout
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 3. Async Node: Model Engine
async def llm_call(state: MessagesState):
    """LLM tracks messages and determines intent pipelines asynchronously."""
    system_instruction = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )
    
    # Prepend the system instructions safely for the client run context execution
    # without corrupting or duplicating items into the state history store channel!
    response = await model_with_tools.ainvoke([system_instruction] + state["messages"])
    
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# 4. Async Node: Concurrent Tool Processor
async def tool_node(state: MessagesState):
    """Executes requested tool bindings asynchronously with type assertions."""
    last_message = state["messages"][-1]
    results = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            target_tool = tools_by_name[tool_call["name"]]
            # Call dynamically using async execution loops
            observation = await target_tool.ainvoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
            
    return {"messages": results}

# 5. Routing Logic Conditional Step
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Inspects structural state signatures to find active execution routing plans."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

# 6. Workflow Assembly Framework
workflow = StateGraph(MessagesState)
workflow.add_node("llm_call", llm_call)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "llm_call")
workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tool_node": "tool_node", END: END}
)
workflow.add_edge("tool_node", "llm_call")

# Compile into an executable agent trace asset
agent = workflow.compile()

# 7. Production Wrapper Function for your FastAPI Router Core
async def generate_llm_response(prompt: str) -> dict[str, str]:
    """
    Business Seam: Accepts a raw user string from your REST endpoint, 
    processes it through the compiled graph, and extracts the ultimate response text.
    """
    initial_input = {"messages": [HumanMessage(content=prompt)], "llm_calls": 0}
    
    # Run the execution graph state trace completely using async handlers
    final_state = await agent.ainvoke(initial_input)
    
    # The absolute last element inside our messages array is our terminal answer message
    final_output_message = final_state["messages"][-1]

    # AI Model that was used for the answer
    provider = settings.LLM_PROVIDER.lower()
    
    #return str(final_output_message.content)
    return {
        "text": str(final_output_message.content),
        "provider": provider
    }