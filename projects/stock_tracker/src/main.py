import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.api.router import api_router


# 1. Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log system startup metrics or model verification processes here
    yield
    # Handle graceful service cleanups here

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        lifespan=lifespan
    )

    # 2. Add Time Logging Middleware
    @app.middleware("http")
    async def log_request_time(request: Request, call_next):
        start_time = time.perf_counter()
        
        # Process the request and get the response from the router/endpoint
        response = call_next(request)
        
        # For pure async middleware safety, wait for response execution
        response = await response 
        
        process_time = time.perf_counter() - start_time
        
        # Log the method, path, status code, and execution time
        logger.info(
            f"Method: {request.method} | "
            f"Path: {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {process_time:.4f}s"
        )
        
        # Inject the processing time into response headers (optional but very useful for debugging)
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        
        return response
    
    # Standard Middleware Stack
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    # Attach Endpoints
    app.include_router(api_router)
    
    return app

app = create_app()