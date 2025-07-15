from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from config.settings import APIConfig
from langchain_openai import AzureChatOpenAI

def supervisor_agent(apiConfig:APIConfig,news_agent: create_react_agent, fundamental_agent:create_react_agent, technical_agent:create_react_agent, humorous_news_agent:create_react_agent) -> create_supervisor:
# def supervisor_agent(apiConfig: APIConfig, news_agent: create_react_agent, fundamental_agent: create_react_agent,
#                        technical_agent: create_react_agent) -> create_supervisor:

  model = AzureChatOpenAI(
      azure_deployment=apiConfig.azure_deployment,
      api_version=apiConfig.azure_api_version,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      azure_endpoint=apiConfig.azure_endpoint,
      api_key=apiConfig.azure_subscription_key)

  supervisor = create_supervisor(
      model=model,
      agents=[news_agent, fundamental_agent,technical_agent, humorous_news_agent],
      prompt=(
          "You are a supervisor managing four agents:\n"
          "- a news agent. Assign news-related tasks to this agent. Use this agent for real news and Do not use this agent for funny or humorous news\n"
          "- a fundamental agent. Assign fundamental analysis tasks to this agent.Do not use this agent for funny or humorous news\n"
          "- a technical agent. Assign technical analysis tasks to this agent. Do not use this agent for funny or humorous news\n"
          "- a humorous news agent. Assign any request for humorous or funny news about Dynatrace to this agent\n"
          "- For stock analysis use news agent, fundamental agent and technical agent\n"
          "- Do not answer questions about anything else.\n"
          # "Assign work to one agent at a time, do not call agents in parallel.\n"
          "Do not do any work yourself."
          "After you get the results, send the results to the users"
      ),
      add_handoff_back_messages=True,
      output_mode="full_history",
  )
  return supervisor
