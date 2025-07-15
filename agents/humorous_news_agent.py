import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import tool
from agents.base_agent import BaseAgent
from config.settings import APIConfig, EmbeddingsConfig
from langchain_core.tools import BaseTool
from typing import List
from functools import lru_cache
from core.exceptions import ToolExecutionError
from core.logging_config import setup_logging
import time

logger = setup_logging()

class HumorousNewsAgent(BaseAgent):
  """Agent for fundamental analysis."""

  def __init__(self, apiConfig: APIConfig, embeddingsConfig: EmbeddingsConfig):
    super().__init__(apiConfig, "humorous_news_agent")
    self.apiConfig = apiConfig
    self.embeddingsConfig = embeddingsConfig
    self.create_vector_database()
    self._search_cache = {}

  def get_tools(self) -> List[BaseTool]:
    """Get fundamental analysis tools."""
    return [
      retrieve_rag_data,
    ]

  def get_prompt(self) -> str:
    """Get agent prompt."""
    return (
      """You are a humorous news agent. Your job is answer the question related to funny news about dynatrace.
      Do not attempt to answer the question on your own. Instead use the tool available to you.
      """)

  def create_vector_database(self):
    dirname = os.getcwd()
    filename = os.path.join(dirname, 'fake_news.txt')
    loader = TextLoader(filename)

    docs = loader.load()

    documents = RecursiveCharacterTextSplitter(
      chunk_size=1000, separators=["\n", "\n\n"], chunk_overlap=200
    ).split_documents(docs)

    endpoint = self.apiConfig.azure_endpoint
    embeddings_model_name = self.embeddingsConfig.embeddings_model_name
    embeddings_deployment = self.embeddingsConfig.embeddings_deployment
    embeddings = AzureOpenAIEmbeddings(
      model=embeddings_model_name,
      azure_deployment=embeddings_deployment,
      azure_endpoint=endpoint,
    )
    db = FAISS.from_documents(
      documents=documents,
      embedding=embeddings
    )
    db.save_local("./faiss-db")


  @lru_cache(maxsize=100)
  def _cached_search(self, query: str, max_age_hours: int = 1) -> str:
    """Cached search to avoid duplicate API calls."""
    cache_key = f"{query}_{int(time.time() // (max_age_hours * 3600))}"

    if cache_key in self._search_cache:
      logger.info(f"Using cached result for query: {query}")
      return self._search_cache[cache_key]

    try:
      search_tool = retrieve_rag_data
      result = search_tool.run(query)
      self._search_cache[cache_key] = result
      return result
    except Exception as e:
      logger.error(f"Search failed for query '{query}': {str(e)}")
      raise ToolExecutionError(f"News search failed: {str(e)}")

@tool
def retrieve_rag_data() -> str:
  """Provide funny news about Dynatrace """
  query = "Funny news about Dynatrace"
  prompt = hub.pull("rlm/rag-prompt")

  endpoint = os.environ.get("AZURE_ENDPOINT")
  embeddings_model_name = os.environ.get("AZURE_EMBEDDINGS_MODEL_NAME")
  embeddings_deployment = os.environ.get("AZURE_EMBEDDINGS_MODEL_DEPLOYMENT")
  embeddings = AzureOpenAIEmbeddings(
    model=embeddings_model_name,
    azure_deployment=embeddings_deployment,
    azure_endpoint=endpoint,
  )

  cwd = os.getcwd()
  vectorstore_faiss = FAISS.load_local(cwd + "/faiss-db", embeddings, allow_dangerous_deserialization=True)
  endpoint = os.environ.get("AZURE_ENDPOINT")
  model_name = os.environ.get("AZURE_MODEL_NAME")
  deployment = os.environ.get("AZURE_DEPLOYMENT")
  subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
  api_version = os.environ.get("AZURE_API_VERSION")
  llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    azure_endpoint=endpoint,
    api_key=subscription_key
  )
  qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
      search_type="similarity", search_kwargs={"k": 6}
    ),
    verbose=False,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
  )

  output = qa.invoke(query)
  return output
