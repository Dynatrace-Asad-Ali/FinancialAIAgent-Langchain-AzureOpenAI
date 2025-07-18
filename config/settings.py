"""Configuration management for the Financial AI Agent."""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings."""
    azure_endpoint: str
    azure_deployment: str
    azure_subscription_key: str
    azure_api_version: str
    tavily_api_key: str
    langchain_api_key: Optional[str] = None
    dynatrace_endpoint: Optional[str] = None
    dynatrace_token: Optional[str] = None


@dataclass
class AppConfig:
    """Application configuration settings."""
    timeout_seconds: int = 120
    recursion_limit: int = 100
    default_model: str = "gpt-4o-mini"
    service_name: str = "FinancialAIAgent"

@dataclass()
class EmbeddingsConfig:
    """Configuration settings for vector embeddings"""
    embeddings_model_name: str
    embeddings_deployment: str


def load_config() -> tuple[APIConfig, AppConfig, EmbeddingsConfig]:
    """Load configuration from environment variables."""
    api_config = APIConfig(
      azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
      azure_deployment = os.environ.get("AZURE_DEPLOYMENT"),
      azure_subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY"),
      azure_api_version = os.environ.get("AZURE_API_VERSION"),
      langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
      dynatrace_endpoint=os.getenv("DYNATRACE_EXPORTER_OTLP_ENDPOINT"),
      dynatrace_token=os.getenv("DYNATRACE_API_TOKEN"),
      tavily_api_key=os.getenv("TAVILY_API_KEY")
    )

    app_config = AppConfig(
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "120")),
        recursion_limit=int(os.getenv("RECURSION_LIMIT", "100")),
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    )

    embeddings_config = EmbeddingsConfig(
        embeddings_model_name = os.environ.get("AZURE_EMBEDDINGS_MODEL_NAME"),
        embeddings_deployment = os.environ.get("AZURE_EMBEDDINGS_MODEL_DEPLOYMENT")
    )
    return api_config, app_config, embeddings_config

def validate_config(api_config: APIConfig) -> list[str]:
    """Validate required configuration."""
    errors = []

    if not api_config.azure_subscription_key:
        errors.append("Azure Subscription Key is required")

    return errors
