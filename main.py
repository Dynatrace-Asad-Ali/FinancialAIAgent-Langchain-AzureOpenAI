"""Improved main application with better structure and error handling."""
import streamlit as st
import asyncio
from typing import Optional, Dict, Any
import time

# Local imports
from config.settings import load_config, validate_config
from core.exceptions import FinancialAgentError, ConfigurationError
from core.logging_config import setup_logging
from core.session_manager import SessionManager
from ui.components import (
    render_sidebar, render_chat_history, render_example_queries,
    render_metrics_dashboard, render_error_message
)
from agents.news_agent import NewsAgent
from agents.fundamental_agent import FundamentalAgent
from agents.technical_agent import TechnicalAgent
from agents.humorous_news_agent import HumorousNewsAgent
from agents.supervisor_agent import supervisor_agent
from utils.utils import astream_graph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage


# Initialize logging
logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FinancialAgentApp:
    """Main application class."""

    def __init__(self):
        self.api_config, self.app_config, self.embeddings_config = load_config()
        self.supervisor = None
        self.metrics = {}

    def validate_configuration(self) -> bool:
        """Validate application configuration."""
        errors = validate_config(self.api_config)

        if errors:
            st.error("❌ Configuration Error")
            for error in errors:
                st.error(f"• {error}")

            st.info("Please check your environment variables and restart the application.")
            return False

        return True

    def initialize_agents(self) -> bool:
        """Initialize all agents."""
        try:
            with st.spinner("🔄 Initializing AI agents..."):
                # Initialize individual agents
                news_agent_instance = NewsAgent(self.api_config)
                fundamental_agent_instance = FundamentalAgent(self.api_config)
                technical_agent_instance = TechnicalAgent(self.api_config)
                humorous_agent_instance = HumorousNewsAgent(self.api_config,)

                # Initialize supervisor
                self.supervisor = supervisor_agent(
                    self.api_config,
                    news_agent_instance.agent,
                    fundamental_agent_instance.agent,
                    technical_agent_instance.agent,
                    humorous_agent_instance.agent
                ).compile()

                st.session_state.agent = self.supervisor
                st.session_state.session_initialized = True

                logger.info("All agents initialized successfully")
                return True

        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            render_error_message("Failed to initialize agents", str(e))
            return False

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

    # async def initialize_session(self):
    #     """
    #     Sets the agent to Supervisor Agent
    #
    #     Returns:
    #         bool: Initialization success status
    #     """
    #
    #     st.session_state.agent = self.supervisor
    #     st.session_state.session_initialized = True
    #     return True

    def get_streaming_callback(self, text_placeholder, tool_placeholder):
        """
        Creates a streaming callback function.

        This function creates a callback function to display responses generated from the LLM in real-time.
        It displays text responses and tool call information in separate areas.

        Args:
            text_placeholder: Streamlit component to display text responses
            tool_placeholder: Streamlit component to display tool call information

        Returns:
            callback_func: Streaming callback function
            accumulated_text: List to store accumulated text responses
            accumulated_tool: List to store accumulated tool call information
        """
        accumulated_text = []
        accumulated_tool = []

        def callback_func(message: dict):
            nonlocal accumulated_text, accumulated_tool
            message_content = message.get("content", None)

            if isinstance(message_content, AIMessageChunk):
                content = message_content.content
                # If content is in list form (mainly occurs in Claude models)
                if isinstance(content, list) and len(content) > 0:
                    message_chunk = content[0]
                    # Process text type
                    if message_chunk["type"] == "text":
                        accumulated_text.append(message_chunk["text"])
                        text_placeholder.markdown("".join(accumulated_text))
                    # Process tool use type
                    elif message_chunk["type"] == "tool_use":
                        if "partial_json" in message_chunk:
                            accumulated_tool.append(message_chunk["partial_json"])
                        else:
                            tool_call_chunks = message_content.tool_call_chunks
                            tool_call_chunk = tool_call_chunks[0]
                            accumulated_tool.append(
                                "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                            )
                        with tool_placeholder.expander(
                                "🔧 Tool Call Information", expanded=True
                        ):
                            st.markdown("".join(accumulated_tool))
                # Process if tool_calls attribute exists (mainly occurs in OpenAI models)
                elif (
                        hasattr(message_content, "tool_calls")
                        and message_content.tool_calls
                        and len(message_content.tool_calls[0]["name"]) > 0
                ):
                    tool_call_info = message_content.tool_calls[0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    with tool_placeholder.expander(
                            "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
                # Process if content is a simple string
                elif isinstance(content, str):
                    accumulated_text.append(content)
                    text_placeholder.markdown("".join(accumulated_text))
                # Process if invalid tool call information exists
                elif (
                        hasattr(message_content, "invalid_tool_calls")
                        and message_content.invalid_tool_calls
                ):
                    tool_call_info = message_content.invalid_tool_calls[0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    with tool_placeholder.expander(
                            "🔧 Tool Call Information (Invalid)", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
                # Process if tool_call_chunks attribute exists
                elif (
                        hasattr(message_content, "tool_call_chunks")
                        and message_content.tool_call_chunks
                ):
                    tool_call_chunk = message_content.tool_call_chunks[0]
                    accumulated_tool.append(
                        "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                    )
                    with tool_placeholder.expander(
                            "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
                # Process if tool_calls exists in additional_kwargs (supports various model compatibility)
                elif (
                        hasattr(message_content, "additional_kwargs")
                        and "tool_calls" in message_content.additional_kwargs
                ):
                    tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                    accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                    with tool_placeholder.expander(
                            "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
            # Process if it's a tool message (tool response)
            elif isinstance(message_content, ToolMessage):
                accumulated_tool.append(
                    "\n```json\n" + str(message_content.content) + "\n```\n"
                )
                with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                    st.markdown("".join(accumulated_tool))
            return None

        return callback_func, accumulated_text, accumulated_tool

    async def process_query(self, query, text_placeholder, tool_placeholder, timeout_seconds=60):
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
                    self.get_streaming_callback(text_placeholder, tool_placeholder)
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

    def run(self):
        """Run the main application."""
        # Header
        st.title("📈 Financial AI Agent")

        # Validate configuration
        if not self.validate_configuration():
            return

        # Initialize session
        SessionManager.initialize_session_state()

        # Render sidebar
        # config = render_sidebar()

        # Update session state with new config
        # st.session_state.timeout_seconds = config["timeout"]
        # st.session_state.recursion_limit = config["recursion_limit"]

        # Initialize agents if not done
        if not st.session_state.get("session_initialized"):
            if not self.initialize_agents():
                return

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


if __name__ == "__main__":
    app = FinancialAgentApp()
    app.run()
