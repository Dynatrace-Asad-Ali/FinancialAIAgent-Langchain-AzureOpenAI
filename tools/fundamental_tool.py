from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.tools import tool

def fundamental_tool() -> YahooFinanceNewsTool:
  my_fundamental_tool = YahooFinanceNewsTool()
  return my_fundamental_tool
