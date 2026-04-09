from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from config.settings import APIConfig


def supervisor_agent(apiConfig:APIConfig,news_agent: create_agent, fundamental_agent:create_agent, technical_agent:create_agent, humorous_news_agent:create_agent) -> create_supervisor:
  model = AzureChatOpenAI(
      azure_deployment=apiConfig.azure_deployment,
      api_version=apiConfig.azure_api_version,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      azure_endpoint=apiConfig.azure_endpoint,
      api_key=apiConfig.azure_subscription_key,
      model="gpt-5.1")

  supervisor = create_supervisor(
      model=model,
      agents=[news_agent, fundamental_agent,technical_agent, humorous_news_agent],
      system_prompt=(
          "You are a supervisor managing four specialized agents. You MUST delegate ALL work to agents — never answer directly.\n\n"
          "AGENTS:\n"
          "- news_agent: fetches real-time news. Use for: latest headlines, recent events, company announcements.\n"
          "- fundamental_agent: performs fundamental analysis. Use for: financial health, valuation, earnings, revenue, balance sheet, investment thesis.\n"
          "- technical_agent: performs technical analysis. Use for: price trends, chart patterns, sector comparison, trading signals.\n"
          "- humorous_news_agent: Use ONLY for funny or humorous news requests about Dynatrace.\n\n"
          "ROUTING RULES:\n"
          "- For ANY stock analysis request (e.g. 'analyze AAPL', 'should I buy MSFT'), you MUST call ALL THREE: news_agent, fundamental_agent, and technical_agent.\n"
          "- For news-only requests (e.g. 'latest news on Tesla'), call only news_agent.\n"
          "- For fundamental-only requests (e.g. 'what are Apple earnings'), call only fundamental_agent.\n"
          "- For technical-only requests (e.g. 'chart patterns for NVDA'), call only technical_agent.\n"
          "- For humorous/funny Dynatrace news, call only humorous_news_agent.\n"
          "- Do not answer questions unrelated to finance or Dynatrace humor.\n\n"
          "After collecting results from all required agents, synthesize and present them to the user."
      ),
      add_handoff_back_messages=True,
      output_mode="full_history",
  )
  return supervisor
