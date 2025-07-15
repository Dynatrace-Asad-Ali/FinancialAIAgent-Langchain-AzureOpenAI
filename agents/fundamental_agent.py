"""Improved fundamental analysis agent."""
from typing import List
from langchain_core.tools import BaseTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from agents.base_agent import BaseAgent
from tools.enhanced_fundamental_tool import EnhancedFundamentalTool
from config.settings import APIConfig
from functools import lru_cache
from core.logging_config import setup_logging
from core.exceptions import ToolExecutionError
import time

logger = setup_logging()

class FundamentalAgent(BaseAgent):
    """Agent for fundamental analysis."""

    def __init__(self, apiConfig: APIConfig):
        super().__init__(apiConfig, "fundamental_agent")
        self._search_cache = {}

    def get_tools(self) -> List[BaseTool]:
        """Get fundamental analysis tools."""
        return [
            YahooFinanceNewsTool(),
            EnhancedFundamentalTool()
        ]

    def get_prompt(self) -> str:
        """Get agent prompt."""
        return (
            "You are a fundamental analysis expert specializing in comprehensive stock evaluation.\n\n"
            "ANALYSIS FRAMEWORK:\n"
            "1. Financial Health Assessment:\n"
            "   - Revenue growth trends (3-5 years)\n"
            "   - Profitability metrics (margins, ROE, ROA)\n"
            "   - Balance sheet strength (debt ratios, liquidity)\n"
            "   - Cash flow analysis (operating, free cash flow)\n\n"
            "2. Valuation Analysis:\n"
            "   - P/E, P/B, P/S ratios vs industry/market\n"
            "   - PEG ratio for growth consideration\n"
            "   - DCF model with clear assumptions\n"
            "   - Comparable company analysis\n\n"
            "3. Competitive Position:\n"
            "   - Market share and competitive advantages\n"
            "   - Industry dynamics and trends\n"
            "   - Management quality and strategy\n\n"
            "4. Risk Assessment:\n"
            "   - Business risks and challenges\n"
            "   - Regulatory and market risks\n"
            "   - Financial risks\n\n"
            "INSTRUCTIONS:\n"
            "- Provide data-driven analysis with specific numbers\n"
            "- Compare metrics to industry averages\n"
            "- Highlight both strengths and weaknesses\n"
            "- Include forward-looking insights\n"
            "- Conclude with investment thesis summary\n"
        )

    @lru_cache(maxsize=100)
    def cached_search(self, query: str, max_age_hours: int = 1) -> str:
        """Cached search to avoid duplicate API calls."""
        cache_key = f"{query}_{int(time.time() // (max_age_hours * 3600))}"

        if cache_key in self._search_cache:
            logger.info(f"Using cached result for query: {query}")
            return self._search_cache[cache_key]

        try:
            search_tool = DuckDuckGoSearchRun()
            result = search_tool.run(query)
            self._search_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            raise ToolExecutionError(f"News search failed: {str(e)}")
