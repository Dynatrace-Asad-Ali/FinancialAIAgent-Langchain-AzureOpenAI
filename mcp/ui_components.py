"""UI components for MCP application."""
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Callable
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=model_options.index(st.session_state.get("selected_model", "gpt-4o-mini")),
            help="Choose the AI model"
        )

        # Timeout settings
        timeout = st.slider(
            "Timeout (seconds)",
            min_value=30,
            max_value=300,
            value=st.session_state.get("timeout_seconds", 120),
            step=30
        )

        # Recursion limit
        recursion_limit = st.slider(
            "Recursion Limit",
            min_value=10,
            max_value=200,
            value=st.session_state.get("recursion_limit", 100),
            step=10
        )

        # Session management
        st.subheader("Session Management")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Reset Session", type="secondary"):
                from session_manager import MCPSessionManager
                manager = MCPSessionManager()
                manager.reset_session()
                st.rerun()

        with col2:
            if st.button("Session Info", type="secondary"):
                from session_manager import MCPSessionManager
                manager = MCPSessionManager()
                st.json(manager.get_session_info())

        return {
            "model": selected_model,
            "timeout": timeout,
            "recursion_limit": recursion_limit
        }

def render_chat_history():
    """Render chat history with improved formatting."""
    if not st.session_state.get("history"):
        st.info("💡 Ask about math operations, weather, or Dynatrace monitoring!")

    # Display chat history
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message["content"])
            i += 1

        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

                # Check for tool information
                if (i + 1 < len(st.session_state.history) and
                    st.session_state.history[i + 1]["role"] == "assistant_tool"):
                    with st.expander("🔧 Tool Information", expanded=False):
                        st.code(st.session_state.history[i + 1]["content"], language="json")
                    i += 2
                else:
                    i += 1
        else:
            i += 1

def render_metrics(tool_count: int):
    """Render metrics dashboard."""
    st.subheader("📊 System Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Available Tools", tool_count)

    with col2:
        st.metric("Active Sessions", 1 if st.session_state.get("session_initialized") else 0)

    # Server status
    st.subheader("🌐 Server Status")
    servers = ["Math Server", "Weather Server", "Dynatrace Server"]

    for server in servers:
        status = "🟢 Online" if st.session_state.get("session_initialized") else "🔴 Offline"
        st.write(f"**{server}:** {status}")

def render_error_message(error: str, details: Optional[str] = None):
    """Render error message with details."""
    st.error(error)

    if details:
        with st.expander("Error Details"):
            st.code(details, language="text")


def get_streaming_callback(text_placeholder, tool_placeholder):
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
