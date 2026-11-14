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
    litellm_proxy_url: Optional[str] = None
    litellm_api_key: Optional[str] = None
    litellm_model: Optional[str] = None
    kong_ai_gateway_url: Optional[str] = None
    kong_ai_gateway_model: Optional[str] = None


@dataclass
class AppConfig:
    """Application configuration settings."""
    timeout_seconds: int = 120
    recursion_limit: int = 100
    default_model: str = "gpt-4o-mini"
    service_name: str = "FinancialAIAgent"

def load_config() -> tuple[APIConfig, AppConfig]:
    """Load configuration from environment variables."""
    api_config = APIConfig(
      azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
      azure_deployment = os.environ.get("AZURE_DEPLOYMENT"),
      azure_subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY"),
      azure_api_version = os.environ.get("AZURE_API_VERSION"),
      langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
      dynatrace_endpoint=os.getenv("DYNATRACE_EXPORTER_OTLP_ENDPOINT"),
      dynatrace_token=os.getenv("DYNATRACE_API_TOKEN"),
      tavily_api_key=os.getenv("TAVILY_API_KEY"),
      litellm_proxy_url=os.getenv("LITELLM_PROXY_URL"),
      litellm_api_key=os.getenv("LITELLM_API_KEY"),
      litellm_model=os.getenv("LITELLM_MODEL"),
      kong_ai_gateway_url=os.getenv("KONG_AI_GATEWAY_URL"),
      kong_ai_gateway_model=os.getenv("KONG_AI_GATEWAY_MODEL"),
    )

    app_config = AppConfig(
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "120")),
        recursion_limit=int(os.getenv("RECURSION_LIMIT", "100")),
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    )

    return api_config, app_config

def validate_config(api_config: APIConfig) -> list[str]:
    """Validate required configuration."""
    errors = []

    if not (api_config.kong_ai_gateway_url or api_config.litellm_proxy_url or api_config.azure_subscription_key):
        errors.append("One of KONG_AI_GATEWAY_URL, LITELLM_PROXY_URL or Azure Subscription Key is required")

    return errors
