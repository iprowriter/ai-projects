import logging
import sys
from src.core.config import settings  

# 2. Configure logging to push strictly to stderr (Crucial for Stdio Transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # Ensures logs don't contaminate the stdout JSON-RPC pipe
)
logger = logging.getLogger("mcp-master-orchestrator")

# 3. register stock server
from src.servers.stock_server import mcp as stock_mcp

def main():
    logger.info(f"Booting up {settings.APP_NAME} in {settings.ENVIRONMENT} mode...")
    
    # If you later add more servers (e.g., general_mcp), FastMCP allows you to 
    # merge them or use stock_mcp directly as your runner:
    logger.info("Initializing Stdio JSON-RPC communication channel...")
    stock_mcp.run(transport="stdio")

if __name__ == "__main__":
    main()