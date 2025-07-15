from langgraph.prebuilt import create_react_agent

def add(a: float, b: float):
  """Add two numbers."""
  return a + b


def multiply(a: float, b: float):
  """Multiply two numbers."""
  return a * b


def divide(a: float, b: float):
  """Divide two numbers."""
  return a / b


def math_agent() -> create_react_agent:
  my_math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
      "You are a math agent.\n\n"
      "INSTRUCTIONS:\n"
      "- Assist ONLY with math-related tasks\n"
      "- After you're done with your tasks, respond to the supervisor directly\n"
      "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
  )
  return my_math_agent
