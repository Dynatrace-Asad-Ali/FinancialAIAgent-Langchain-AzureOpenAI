from typing import List
from langchain_core.tools import BaseTool
from agents.base_agent import BaseAgent
from config.settings import APIConfig
from langchain_tavily import TavilySearch


class NewsAgent(BaseAgent):
    """News agent for fetching latest news."""

    def __init__(self, apiConfig: APIConfig):
        super().__init__(apiConfig, "news_agent")

    def get_tools(self) -> List[BaseTool]:
        return [TavilySearch(max_results=5, topic="news")]

    def get_prompt(self) -> str:
        return (
            "You are a news agent that helps users find the latest news.\n\n"
            "INSTRUCTIONS:\n"
            "- Focus on recent, relevant news articles\n"
            "- Provide concise summaries with key information\n"
            "- Include publication dates when available\n"
            "- Verify information from multiple sources when possible\n"
            "- If no recent news is found, clearly state this\n"
            "- Limit responses to 4 most relevant news items\n"
            "- Always cite sources with URLs when available\n"
        )
