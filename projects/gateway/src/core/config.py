from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_ENV = Path(__file__).parents[4] / ".env"
load_dotenv(_ROOT_ENV)  # populates os.environ for LangSmith and other libs

class Settings(BaseSettings):
    APP_NAME: str = "AI Core Gateway"
    ENVIRONMENT: str = "development"

    # API Secret bindings (Loaded straight from root .env)
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    OLLAMA_MODEL: str = "llama3.2:latest" # "gemma3:270m"
    LLM_PROVIDER: str  = "ollama" # Options: 'gemini' | 'ollama'


    model_config = SettingsConfigDict(
        env_file=str(_ROOT_ENV),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()