import asyncio
import logging
from typing import Any, List, Dict
import httpx
from langchain.tools import tool
from src.database.supabase import fetch_all_portfolios

logger = logging.getLogger("uvicorn.error")

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


@tool
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


@tool
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
# 2. DETERMINISTIC MATH UTILITY (Hidden from LLM)
# ==========================================

def calculate_portfolio_worth(holdings: List[Dict[str, Any]], live_prices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A pure Python calculation engine. It takes raw database positions and raw
    market data, combines them, and calculates precise portfolio valuations.
    This is kept out of the tool layer to enforce clean segregation of concerns.
    """
    prices = {p["ticker"]: p for p in live_prices}
    positions = []
    total_value = 0.0

    for holding in holdings:
        ticker = holding["ticker_symbol"]
        shares = float(holding.get("amount_held", 0))
        
        market_data = prices.get(ticker, {})
        price = market_data.get("price")
        currency = market_data.get("currency", "USD")
        
        value = round(shares * price, 2) if price is not None else None
        if value is not None:
            total_value += value
            
        positions.append({
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": value,
            "currency": currency,
        })

    return {
        "positions": positions,
        "total_value": round(total_value, 2),
        "currency": "USD",
    }


# @tool
# async def calculate_portfolio_worth(user_id: int) -> Dict[str, Any]:
#     """
#     Fetches a user's portfolio from the database and calculates its current total
#     worth by multiplying each holding's shares by the live market price.

#     Args:
#         user_id (int): The unique identification integer of the user.

#     Returns:
#         Dict[str, Any]: Contains:
#             - 'positions': per-stock breakdown with ticker, shares, price, value, and currency.
#             - 'total_value': sum of all positions in USD.
#             - 'currency': "USD".
#     """
#     holdings = fetch_all_portfolios(user_id=user_id)
#     tickers = [h["ticker_symbol"] for h in holdings]

#     async def _fetch_one(client: httpx.AsyncClient, ticker: str) -> Dict[str, Any]:
#         try:
#             r = await client.get(
#                 _YAHOO_CHART_URL.format(ticker=ticker),
#                 params={"interval": "1d", "range": "1d"},
#             )
#             r.raise_for_status()
#             meta = r.json()["chart"]["result"][0]["meta"]
#             return {
#                 "ticker": ticker,
#                 "price": meta.get("regularMarketPrice"),
#                 "currency": meta.get("currency", "USD"),
#             }
#         except Exception as e:
#             logger.error(f"Price fetch failed for {ticker}: {e}")
#             return {"ticker": ticker, "price": None, "currency": None}

#     async with httpx.AsyncClient(headers=_YAHOO_HEADERS, timeout=10) as client:
#         price_data = await asyncio.gather(*[_fetch_one(client, t) for t in tickers])

#     prices = {p["ticker"]: p for p in price_data}
#     positions = []
#     total_value = 0.0

#     for holding in holdings:
#         ticker = holding["ticker_symbol"]
#         shares = float(holding.get("amount_held", 0))
#         price = prices.get(ticker, {}).get("price")
#         currency = prices.get(ticker, {}).get("currency", "USD")
#         value = round(shares * price, 2) if price is not None else None
#         if value is not None:
#             total_value += value
#         positions.append({
#             "ticker": ticker,
#             "shares": shares,
#             "price": price,
#             "value": value,
#             "currency": currency,
#         })

#     return {
#         "positions": positions,
#         "total_value": round(total_value, 2),
#         "currency": "USD",
#     }