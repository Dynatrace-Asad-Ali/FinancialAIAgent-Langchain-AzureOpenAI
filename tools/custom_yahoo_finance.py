from typing import Iterable, Optional, Type
import yfinance
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError
from langchain_community.document_loaders.web_base import WebBaseLoader

class YahooFinanceNewsInput(BaseModel):
    """Input for the YahooFinanceNews tool."""
    query: str = Field(description="company ticker query to look up")

class CustomYahooFinanceNewsTool(BaseTool):
    """Tool that searches financial news on Yahoo Finance. Bypasses .isin check to prevent curl timeout."""
    
    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10
    args_schema: Type[BaseModel] = YahooFinanceNewsInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        company = yfinance.Ticker(query)
        
        # Bypass .isin check to prevent curl_cffi timeout.
        links = []
        try:
            news = company.news
            links = [
                n["content"]["canonicalUrl"]["url"]
                for n in news
                if "content" in n and "contentType" in n["content"] and n["content"]["contentType"] == "STORY"
            ]
        except Exception as e:
            return f"Company ticker {query} not found. Error: {str(e)}"

        if not links:
            return f"No news found for company that searched with {query} ticker."

        try:
            loader = WebBaseLoader(web_paths=links)
            docs = loader.load()
            result = self._format_results(docs, query)
            if not result:
                return f"No news found for company that searched with {query} ticker."
            return result
        except Exception as e:
            return f"Error loading news documents for {query}: {str(e)}"

    @staticmethod
    def _format_results(docs: Iterable[Document], query: str) -> str:
        doc_strings = [
            "\n".join([doc.metadata.get("title", ""), doc.metadata.get("description", "")])
            for doc in docs
            if query in doc.metadata.get("description", "")
            or query in doc.metadata.get("title", "")
        ]
        return "\n\n".join(doc_strings)
