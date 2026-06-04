
from src.rag.retrieval import vector_store
from langchain.tools import tool

@tool
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