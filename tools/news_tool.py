from dotenv import load_dotenv
import requests
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


def news_search() -> DuckDuckGoSearchRun:
  my_news_search = DuckDuckGoSearchRun()
  return my_news_search


def weather_lookup(location: str) -> str:
  """Find the weather of a location."""
  r = requests.get(
      'https://api.weatherapi.com/v1/current.json?q=' + location + '&key=redacted')
  return r.json()
