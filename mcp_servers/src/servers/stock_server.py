import asyncio
import logging
import httpx
from fastmcp import FastMCP
from typing import List, Dict, Any
from src.database.supabase import fetch_all_portfolios
from src.rag.retrieval import vector_store

mcp = FastMCP("Stock")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stock-mcp-server")

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}

# ==========================================
# 1. Retrieve Data from my Supabase database
# ==========================================
@mcp.tool()
async def get_user_portfolio(user_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves the complete investment portfolio for a given user, returning
    assets held and their respective quantities.

    Args:
        user_id (int): The unique identification integer of the user.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing 'ticker_symbol' and 'amount_held'.
    """
    try:
        response = fetch_all_portfolios(user_id=user_id)
        return response if response else []
    except Exception as e:
        logger.error(f"Supabase database failure for user {user_id}: {str(e)}")
        return []
    

# ==========================================
# 2. Retrieve Data from an external API
# ==========================================
@mcp.tool()
async def get_live_prices(ticker_symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches the current live market price for one or more stock ticker symbols.
    Use this to find out how much each holding is currently worth.

    Args:
        ticker_symbols (List[str]): A list of stock ticker symbols, e.g. ["AAPL", "TSLA"].

    Returns:
        List[Dict[str, Any]]: Each item contains 'ticker', 'price', and 'currency'.
        If a ticker cannot be found, 'price' will be null.
    """
    async def _fetch_one(client: httpx.AsyncClient, ticker: str) -> Dict[str, Any]:
        try:
            r = await client.get(
                _YAHOO_CHART_URL.format(ticker=ticker),
                params={"interval": "1d", "range": "1d"},
            )
            r.raise_for_status()
            meta = r.json()["chart"]["result"][0]["meta"]
            return {
                "ticker": ticker,
                "price": meta.get("regularMarketPrice"),
                "currency": meta.get("currency", "USD"),
            }
        except Exception as e:
            logger.error(f"Price fetch failed for {ticker}: {e}")
            return {"ticker": ticker, "price": None, "currency": None}

    async with httpx.AsyncClient(headers=_YAHOO_HEADERS, timeout=10) as client:
        return await asyncio.gather(*[_fetch_one(client, t) for t in ticker_symbols])
    
# ==========================================
# 3. Retrieve data from Pinecone Vector DB Embeddings
# ==========================================
@mcp.tool()
def query_apple_financials_report(query:str) -> str:
    """
    Searches the Apple (AAPL) financial document to retrieve context chunks 
    regarding balance sheets, revenue, guidance, and risks.
    """
    # Retrieve the top 3 most relevant textual document pieces
    results = vector_store.similarity_search(query, k=3)

    # Compile chunks into a single text block back to LangGraph agent node loop
    context_block = "\n--\n".join([doc.page_content for doc in results])
    return context_block





if __name__ == "__main__":
    mcp.run(transport="stdio")