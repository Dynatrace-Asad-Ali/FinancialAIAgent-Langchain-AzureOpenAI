"""MCP application with better structure and error handling."""
import streamlit as st
import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Core imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage


# Local imports
from utils import astream_graph, random_uuid
from ui_components import (
    render_sidebar, render_chat_history, render_metrics,
    render_error_message, get_streaming_callback
)
from config_manager import ConfigManager
from session_manager import MCPSessionManager

load_dotenv()

# Initialize tracing if configured
try:
    from traceloop.sdk import Traceloop
    if os.environ.get("DYNATRACE_API_TOKEN"):
        headers = {"Authorization": f"Api-Token {os.environ.get('DYNATRACE_API_TOKEN')}"}
        Traceloop.init(
            app_name="MCPAgent",
            api_endpoint=os.environ.get("DYNATRACE_EXPORTER_OTLP_ENDPOINT"),
            headers=headers,
            disable_batch=True
        )
except ImportError:
    st.warning("Traceloop not available. Tracing disabled.")

# Page configuration
st.set_page_config(
    page_title="MCP Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MCPApplication:
    """Main MCP application class."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.session_manager = MCPSessionManager()
        self.metrics = {}

    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        defaults = {
            "session_initialized": False,
            "agent": None,
            "history": [],
            "mcp_client": None,
            "timeout_seconds": 120,
            "selected_model": "gpt-4o-mini",
            "recursion_limit": 100,
            "thread_id": random_uuid(),
            "event_loop": None,
            "tool_count": 0,
            "pending_mcp_config": self.config_manager.load_config()
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Initialize event loop
        if st.session_state.event_loop is None:
            loop = asyncio.new_event_loop()
            st.session_state.event_loop = loop
            asyncio.set_event_loop(loop)

    async def initialize_mcp_session(self, mcp_config: Optional[Dict] = None) -> bool:
        """Initialize MCP session with proper error handling."""
        try:
            with st.spinner("🔄 Connecting to MCP servers..."):
                # Cleanup existing client
                await self.session_manager.cleanup_client()

                if mcp_config is None:
                    mcp_config = self.config_manager.load_config()

                # Validate configuration
                if not self.config_manager.validate_config(mcp_config):
                    st.error("❌ Invalid MCP configuration")
                    return False

                # Initialize client
                client = MultiServerMCPClient(mcp_config)
                tools = await client.get_tools()

                st.session_state.tool_count = len(tools)
                st.session_state.mcp_client = client

                endpoint = os.environ.get("AZURE_ENDPOINT")
                deployment = os.environ.get("AZURE_DEPLOYMENT")
                subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
                api_version = os.environ.get("AZURE_API_VERSION")

                # Initialize model
                model = AzureChatOpenAI(
                  azure_deployment=deployment,
                  api_version=api_version,
                  max_tokens=None,
                  timeout=None,
                  max_retries=2,
                  azure_endpoint=endpoint,
                  api_key=subscription_key
                )

                # Create agent
                agent = create_react_agent(
                    model,
                    tools,
                    checkpointer=MemorySaver(),
                    prompt=self._get_system_prompt(),
                )

                st.session_state.agent = agent
                st.session_state.session_initialized = True

                st.success(f"✅ Connected to {len(tools)} tools across MCP servers")
                return True

        except Exception as e:
            st.error(f"❌ Failed to initialize MCP session: {str(e)}")
            return False

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """
You are an intelligent AI assistant with access to multiple specialized tools through MCP servers.

CAPABILITIES:
- Math operations and calculations
- Weather information lookup
- Dynatrace monitoring and observability queries
- General problem-solving and analysis

INSTRUCTIONS:
1. Analyze the user's question carefully
2. Select the most appropriate tool(s) to answer the question
3. Provide clear, professional, and helpful responses
4. If using tools, base your answer primarily on the tool output
5. Include sources when applicable (URLs or references)
6. Be concise but comprehensive in your responses

RESPONSE FORMAT:
- Direct answer to the question
- Supporting details from tools
- Sources (if applicable)

Remember: Only use the tools provided. Do not make up information.
"""

    async def process_query(self,query, text_placeholder, tool_placeholder, timeout_seconds=60):
      """
      Processes user questions and generates responses.

      This function passes the user's question to the agent and streams the response in real-time.
      Returns a timeout error if the response is not completed within the specified time.

      Args:
          query: Text of the question entered by the user
          text_placeholder: Streamlit component to display text responses
          tool_placeholder: Streamlit component to display tool call information
          timeout_seconds: Response generation time limit (seconds)

      Returns:
          response: Agent's response object
          final_text: Final text response
          final_tool: Final tool call information
      """
      try:
        if st.session_state.agent:
          streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
            get_streaming_callback(text_placeholder, tool_placeholder)
          )
          try:
            response = await asyncio.wait_for(
              astream_graph(
                st.session_state.agent,
                {"messages": [HumanMessage(content=query)]},
                callback=streaming_callback,
                config=RunnableConfig(
                  recursion_limit=st.session_state.recursion_limit,
                  thread_id=st.session_state.thread_id,
                ),
              ),
              timeout=timeout_seconds,
            )
          except asyncio.TimeoutError:
            error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
            return {"error": error_msg}, error_msg, ""

          final_text = "".join(accumulated_text_obj)
          final_tool = "".join(accumulated_tool_obj)
          return response, final_text, final_tool
        else:
          return (
            {"error": "🚫 Agent has not been initialized."},
            "🚫 Agent has not been initialized.",
            "",
          )
      except Exception as e:
        import traceback

        error_msg = f"❌ Error occurred during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

    def print_message(self):
      """
      Displays chat history on the screen.

      Distinguishes between user and assistant messages on the screen,
      and displays tool call information within the assistant message container.
      """
      i = 0
      while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
          st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
          i += 1
        elif message["role"] == "assistant":
          # Create assistant message container
          with st.chat_message("assistant", avatar="🤖"):
            # Display assistant message content
            st.markdown(message["content"])

            # Check if the next message is tool call information
            if (
              i + 1 < len(st.session_state.history)
              and st.session_state.history[i + 1]["role"] == "assistant_tool"
            ):
              # Display tool call information in the same container as an expander
              with st.expander("🔧 Tool Call Information", expanded=False):
                st.markdown(st.session_state.history[i + 1]["content"])
              i += 2  # Increment by 2 as we processed two messages together
            else:
              i += 1  # Increment by 1 as we only processed a regular message
        else:
          # Skip assistant_tool messages as they are handled above
          i += 1

    def run(self):
        """Run the main application."""
        # Header
        st.title("🤖 MCP Multi-Agent System")
        # Initialize session
        self.initialize_session_state()

        # Initialize MCP session if needed
        if not st.session_state.get("session_initialized"):
            success = st.session_state.event_loop.run_until_complete(
                self.initialize_mcp_session(st.session_state.pending_mcp_config)
            )
            if not success:
                st.stop()

        self.print_message()
        user_query = st.chat_input("💬 Enter your question")
        if user_query:
          if st.session_state.session_initialized:
            st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
            with st.chat_message("assistant", avatar="🤖"):
              tool_placeholder = st.empty()
              text_placeholder = st.empty()
              resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                  self.process_query(
                    user_query,
                    text_placeholder,
                    tool_placeholder,
                    st.session_state.timeout_seconds,
                  )
                )
              )
            if "error" in resp:
              st.error(resp["error"])
            else:
              st.session_state.history.append({"role": "user", "content": user_query})
              st.session_state.history.append(
                {"role": "assistant", "content": final_text}
              )
              if final_tool.strip():
                st.session_state.history.append(
                  {"role": "assistant_tool", "content": final_tool}
                )
              st.rerun()

        # Main content
        # col1 = st.columns([1])

        # with col1:
            # Chat interface
            # render_chat_history()

            # Chat input
            # user_query = st.chat_input("💬 Ask about math, weather, or Dynatrace...")
            #
            # if user_query:
                # Add user message
                # st.session_state.history.append({"role": "user", "content": user_query})

                # Display user message
                # with st.chat_message("user", avatar="🧑‍💻"):
                #     st.markdown(user_query)

                # Process query
                # result = st.session_state.event_loop.run_until_complete(
                #     self.process_query(user_query)
                # )

                # if "error" in result:
                #     render_error_message(result["error"])
                # else:
                    # Add assistant response to history
                    # st.session_state.history.append({
                    #     "role": "assistant",
                    #     "content": result["final_text"]
                    # })

                    # if result["final_tool"].strip():
                    #     st.session_state.history.append({
                    #         "role": "assistant_tool",
                    #         "content": result["final_tool"]
                    #     })

                # st.rerun()

if __name__ == "__main__":
    app = MCPApplication()
    app.run()
