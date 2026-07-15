# FinancialAIAgent-Langchain
This sample application does stock analysis by conducting news search, fundamental review and technical analysis.
This app is built on a Supervisor with multiple agents architecture as described [here](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb).
The agents use Azure OpenAI, TavilySearch for news, and Yahoo Finance for fundamental and technical analysis.

### API Keys

#### Required
* `AZURE_ENDPOINT` — Azure OpenAI resource endpoint
* `AZURE_DEPLOYMENT` — Azure OpenAI deployment name
* `AZURE_SUBSCRIPTION_KEY` — Azure OpenAI API key
* `AZURE_API_VERSION` — Azure OpenAI API version (e.g. `2024-02-01`)
* `TAVILY_API_KEY` — Tavily search API key (used by the news agent)

#### Optional — Observability (Dynatrace via Traceloop)
* `DYNATRACE_EXPORTER_OTLP_ENDPOINT` — e.g. `<tenant>.live.dynatrace.com/api/v2/otlp`
* `DYNATRACE_API_TOKEN` — Dynatrace API token with ingest scope
* `OTEL_SERVICE_NAME` — Service name shown in traces (defaults to `FinancialAIAgent`)

### Requirements
* Python 3.9 or higher

### How to run

#### On Desktop
```bash
cp .env.template .env
# Edit .env with your keys

python -m venv financialagent
source financialagent/bin/activate
pip install -r requirements.txt

streamlit run main.py
```

#### Codespaces
The devcontainer fetches Azure and OTel secrets from a secret server at startup using a password you provide. Set the following as a Codespaces secret in your GitHub repository settings:
* `WORKSHOP_PASSWORD` — password for the secret server

Dynatrace and Tavily keys must also be set as Codespaces secrets:
* `DYNATRACE_EXPORTER_OTLP_ENDPOINT`
* `DYNATRACE_API_TOKEN`
* `TAVILY_API_KEY`

Once secrets are configured, open the repo in Codespaces and run:
```bash
./run.sh
```

### Suggested Chat Commands
* Analyze AAPL stock
* Give me latest news on Nvidia
* Conduct fundamental analysis on DIS
* What are the technical indicators for MSFT?
