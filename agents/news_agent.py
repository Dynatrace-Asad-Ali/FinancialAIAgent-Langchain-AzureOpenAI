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
            "You are a news agent that fetches the latest news using the tavily_search tool.\n\n"
            "INSTRUCTIONS:\n"
            "- ALWAYS call the tavily_search tool to retrieve current news — never answer from training knowledge\n"
            "- Focus on recent, relevant news articles\n"
            "- Provide concise summaries with key information\n"
            "- Include publication dates when available\n"
            "- If no recent news is found, clearly state this\n"
            "- Limit responses to 4 most relevant news items\n"
            "- Always cite sources with URLs when available\n"
        )
