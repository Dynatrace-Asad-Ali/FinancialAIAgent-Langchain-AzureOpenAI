"""Base agent class with common functionality."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from core.exceptions import AgentInitializationError
from core.logging_config import setup_logging
from config.settings import APIConfig
from langchain_openai import AzureChatOpenAI

logger = setup_logging()

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, apiConfig: APIConfig, name: str):
      model = AzureChatOpenAI(
        azure_deployment=apiConfig.azure_deployment,
        api_version=apiConfig.azure_api_version,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        azure_endpoint=apiConfig.azure_endpoint,
        api_key=apiConfig.azure_subscription_key
      )
      self.model = model
      self.name = name
      self._agent = None
      self._tools = []

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get tools for this agent."""
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Get prompt for this agent."""
        pass

    def initialize(self) -> create_react_agent:
        """Initialize the agent."""
        try:
            self._tools = self.get_tools()
            prompt = self.get_prompt()

            self._agent = create_react_agent(
                model=self.model,
                tools=self._tools,
                prompt=prompt,
                name=self.name
            )

            logger.info(f"Initialized {self.name} with {len(self._tools)} tools")
            return self._agent

        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {str(e)}")
            raise AgentInitializationError(f"Failed to initialize {self.name}: {str(e)}")

    @property
    def agent(self) -> create_react_agent:
        """Get the initialized agent."""
        if self._agent is None:
            self._agent = self.initialize()
        return self._agent
