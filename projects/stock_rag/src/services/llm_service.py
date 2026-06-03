from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator
import asyncio
import logging
import uuid

from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.core.config import settings
from src.tools.analytics import get_user_portfolio, get_live_prices, calculate_portfolio_worth
from src.tools.retrieved_embeddings import query_apple_financials_report
import json


#logger
logger = logging.getLogger("uvicorn.error")

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
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two integers a and b."""
#     return a * b

# @tool
# def add(a: int, b: int) -> int:
#     """Add two integers a and b."""
#     return a + b


tools = [get_user_portfolio, get_live_prices, query_apple_financials_report]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to the initialized base client
model_with_tools = raw_model.bind_tools(tools)

# 2. Define Clean State Layout
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    portfolio_data: list  # replaced each write; holds last get_user_portfolio result
    price_data: list      # replaced each write; holds last get_live_prices result

# 3. Async Node: Model Engine
async def llm_call(state: MessagesState):
    """LLM tracks messages and determines intent pipelines asynchronously."""
    system_instruction = SystemMessage(
        content="""
        You are an Elite Portfolio Analytics Agent.
        You have direct access to database search tools and live pricing tickers.

        To answer questions about portfolio worth or valuation:
        1. Call get_user_portfolio with the user_id to fetch holdings.
        2. Call get_live_prices with the ticker symbols to fetch current prices.
        3. The system will compute the valuation deterministically and inject a
           [CALCULATION_RESULT] block into the conversation.
        4. When you see a [CALCULATION_RESULT] block, format it into a clear,
           human-readable summary. Do NOT recalculate — use the numbers as given.
        5. Call query_apple_financials_report to answer questions regarding APPLE or AAPL financials. 
        6. Do not assume APPLE or AAPL financials if it is not in the embeddings

        Current Active Context: user_id=1.
        """
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
    """Executes requested tool bindings concurrently and captures data for the aggregator."""
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    observations: dict = {}

    async def _run_tool(tool_call):
        target_tool = tools_by_name[tool_call["name"]]
        observation = await target_tool.ainvoke(tool_call["args"])
        observations[tool_call["name"]] = observation
        return ToolMessage(content=str(observation), tool_call_id=tool_call["id"])

    results = await asyncio.gather(*[_run_tool(call) for call in last_message.tool_calls])

    updates: dict = {"messages": list(results)}
    if "get_user_portfolio" in observations:
        updates["portfolio_data"] = observations["get_user_portfolio"]
    if "get_live_prices" in observations:
        updates["price_data"] = observations["get_live_prices"]
    return updates


# 4b. Deterministic Aggregator Node
def aggregator_node(state: MessagesState):
    """Runs calculate_portfolio_worth() when both data sources are present, then clears them."""
    portfolio = state.get("portfolio_data", [])
    prices = state.get("price_data", [])

    if not portfolio or not prices:
        return {}

    worth = calculate_portfolio_worth(portfolio, prices)

    return {
        "messages": [HumanMessage(content=f"[CALCULATION_RESULT]\n{json.dumps(worth, indent=2)}")],
        "portfolio_data": [],
        "price_data": [],
    }

# 5. Routing Logic Conditional Step
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Inspects structural state signatures to find active execution routing plans."""
    # if condition to prevent endless execution loops
    if state.get("llm_calls", 0) > 5:
        logger.warning("Agent execution loop short-circuited due to safety thresholds.")
        return END
    
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

# 6. Workflow Assembly Framework
workflow = StateGraph(MessagesState)
workflow.add_node("llm_call", llm_call)
workflow.add_node("tool_node", tool_node)
workflow.add_node("aggregator_node", aggregator_node)

workflow.add_edge(START, "llm_call")
workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tool_node": "tool_node", END: END}
)
workflow.add_edge("tool_node", "aggregator_node")
workflow.add_edge("aggregator_node", "llm_call")

# Memory
memory = MemorySaver()

# Compile into an executable agent trace asset
agent = workflow.compile(checkpointer=memory)

# 7. Production Wrapper Function for your FastAPI Router Core
async def generate_llm_response(prompt: str, thread_id: str | None = None) -> dict[str, str]:
    """
    Production-ready wrapper. Pass thread_id to continue a session; omit for a fresh one.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    initial_input = {
        "messages": [HumanMessage(content=prompt)],
        "llm_calls": 0,
        "portfolio_data": [],
        "price_data": [],
    }
    
    # Run the execution graph state trace completely using async handlers
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await agent.ainvoke(initial_input, config=config)
    
    # The absolute last element inside our messages array is our terminal answer message
    final_output_message = final_state["messages"][-1]

    # AI Model that was used for the answer
    provider = settings.LLM_PROVIDER.lower()
    
    #return str(final_output_message.content)
    return {
        "text": str(final_output_message.content),
        "provider": provider
    }