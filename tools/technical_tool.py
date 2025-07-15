from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.tools import tool

def technical_tool() -> YahooFinanceNewsTool:
  my_technical_tool = YahooFinanceNewsTool()
  return my_technical_tool
