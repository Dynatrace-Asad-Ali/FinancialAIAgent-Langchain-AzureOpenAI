"""Improved fundamental analysis agent."""
from typing import List
from langchain_core.tools import BaseTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from agents.base_agent import BaseAgent
from tools.enhanced_fundamental_tool import EnhancedFundamentalTool
from config.settings import APIConfig

class TechnicalAgent(BaseAgent):
    """Agent for Technical analysis."""

    def __init__(self, apiConfig: APIConfig):
        super().__init__(apiConfig, "technical_agent")

    def get_tools(self) -> List[BaseTool]:
        """Get fundamental analysis tools."""
        return [
          YahooFinanceNewsTool()
        ]

    def get_prompt(self) -> str:
        """Get agent prompt."""
        return (
            "You are a technical analysis agent that helps users analyze stock prices and trends for a given stock {stock}.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
            "Perform technical analysis on the stock. Include:\n"
                # "Moving Averages (1 Year): 50-day & 200-day, with crossovers.\n"
                # "Support & Resistance: 3 levels each, with significance.\n"
                # "Volume Analysis (3 Months): Trends and anomalies.\n"
                # "RSI & MACD: Compute and interpret signals.\n"
                # "Fibonacci Levels: Calculate and analyze.\n"
                "Chart Patterns (6 Months): Identify 3 key patterns.\n"
                "Sector Comparison: Contrast with sector averages.\n"
            "Get the data from the tool and pass it on to the supervisor"
        )


