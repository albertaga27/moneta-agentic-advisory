# Moneta - an AI-Agentic Assistant for Insurance, Banking and Energy sector

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/albertaga27/aoai-fsi-empowering-advisory-agentic) [![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/albertaga27/aoai-fsi-empowering-advisory-agentic)

Moneta is an AI-powered assistant designed to empower insurance and banking advisors. This Solution Accelerator provides a chat interface where advisors can interact with various AI agents specialized in different domains such as insurance policies, CRM, product information, funds, CIO insights, and news.

## 🚀 Agent Framework & Azure AI Foundry

Moneta uses the **Microsoft Agent Framework** to orchestrate **native Azure AI Foundry agents**:

* [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - Multi-agent orchestration with HandoffBuilder pattern
* [Azure AI Foundry](https://ai.azure.com/) - Native hosted agents with versioning support

### Key Architecture Features

- **Native Foundry Agents**: Agents are hosted in Azure AI Foundry with automatic versioning and reuse
- **HandoffBuilder Pattern**: Coordinator agent routes requests to specialist agents (CRM, CIO, Funds, News)
- **Conversation Memory**: Full conversation history is maintained across API calls via CosmosDB
- **OpenTelemetry Tracing**: Built-in observability with Azure Application Insights integration
- **Session-Level Tracing**: Custom spans for conversation turns and agent handoffs

> **⚠️ Important Note on Handoff Tools**
> 
> Foundry-hosted agents (`AzureAIClient`) don't properly support tool calling with synthesized handoff tools - the model outputs tool names as text instead of calling them as functions. For this reason, the orchestrator workflow uses `AzureOpenAIChatClient` (Azure OpenAI) for reliable handoff execution, while Foundry is still used for agent persistence and versioning when using the `--foundry --new` flags.

## Prerequisites

* Docker
* [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager
* Python 3.12
* Azure CLI (logged in)
* Azure AI Foundry project (for hosted agents)

## Features

- **Microsoft Agent Framework**: Multi-agent orchestration with HandoffBuilder pattern
- **Azure AI Foundry Integration**: Native hosted agents with versioning support
- Multi-Use Case Support: Switch between insurance, banking and energy use cases
- Agent Collaboration: Coordinator routes to specialists who collaborate to provide answers
- Azure AD Authentication: Secure login with Microsoft Azure Active Directory
- Conversation History: Access and continue previous conversations with full context

## Implementation Details
- Python 3.12 or higher
- **Microsoft Agent Framework** with HandoffBuilder for multi-agent orchestration
- **Azure AI Foundry** for native hosted agents with versioning
- Streamlit (frontend app - chatGPT style with conversation segregation and memory)
- FastAPI (backend API with async support)
- Microsoft Authentication Library (MSAL - if using authentication - optional)
- Azure AD application registration (if using authentication - optional)
- An Azure Container App hosting backend API endpoint
- CosmosDB to store user conversations and history
- Azure Application Insights for OpenTelemetry tracing

## Use Cases

### Insurance

- `CRM`: simulate fetching clients information from a CRM (DB, third-party API etc)
- `Policies RAG`: vector search with AI Search on various public available policy documents (product information)
- `Responder`: collects previous agents replies and respond to the user

### Banking 

Uses the **HandoffBuilder** pattern with a coordinator that routes to specialist agents:

- `Coordinator`: Routes user requests to appropriate specialist agents
- `CRM Agent`: Fetches client information and portfolio data from CRM (simulated)
- `Funds Agent`: Vector search with AI Search on funds and ETF factsheets
- `CIO Agent`: Vector search with AI Search on in-house investment views and recommendations
- `News Agent`: RSS online feed search on stock news for portfolio positions

All agents are hosted as **native Azure AI Foundry agents** with automatic versioning.

### Energy

- `News`: energy realted news RSS
- `Electricity`: Swiss electricity grid consumption and production data
- `Weather`: simple weather forecast api search
- `Insights`: Analyze other agents information input and provide insights and classification 
- `Responder`: collects previous agents replies and respond to the user


## Project structure

- src
  - backend
    - foundry
      - agents
        - banking # Foundry agent definitions and functions
          - cio/ # CIO agent with AI Search functions
          - crm/ # CRM agent with client data functions
          - funds/ # Funds agent with AI Search functions
          - news/ # News agent with RSS feed functions
        - insurance # agents files (legacy)
        - energy # agents files (legacy)
      - orchestrators
        - foundry_banking_orchestrator.py # Main orchestrator with HandoffBuilder
    - app.py # FastAPI backend exposing API

  - frontend
    - app.py # Streamlit app

  - data
    - ai-search-index
      - cio-index
      - funds-index
      - ins-index
    - customer-profile

- infra
  - bicep files
  - infra modules


### Azure deployment (automated)

To configure, follow these steps:

1. Make sure you AZ CLI is logged in in the right tenant. Optionally:

    ```shell
    az login --tenant your_tenant.onmicrosoft.com
    ```

1. Create a new azd environment:

    ```shell
    azd env new
    ```

    This will create a folder under `.azure/` in your project to store the configuration for this deployment. You may have multiple azd environments if desired.

1. Set the `AZURE_AUTH_TENANT_ID` azd environment variable to the tenant ID you want to use for Entra authentication:

    ```shell
    azd env set AZURE_AUTH_TENANT_ID $(az account show --query tenantId -o tsv)
    ```

1. Login to the azd CLI with the Entra tenant ID:

    ```shell
    azd auth login --tenant-id $(azd env get-value AZURE_AUTH_TENANT_ID)
    ```

1. Proceed with AZD deployment:

    ```shell
    azd up
    ```

### Data indexing (optional)

Demo data is NOT loaded with the `azd up` process automatically.

If you want to provide AI Search services and load demo data into indexes for the banking and insurance agents
you can do it by running:
```shell
azd hooks run postdeploy
```

Indexes are sourced from 'src/data/ai-search-index' folder.
Each subfolder of the data folder will be a seperate index. 

Customer profiles are sourced from 'src/data/customer-profiles'.
Each subfolder of the data folder will be get a seperate index. 

**OBS!** If you deploy from WSL mounted path, the postdeploy data init might fail. Please consider rerunning from WSL native path location.


### Running the App locally - BACKEND

The python project is managed by `pyproject.toml` and the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).
Install uv prior executing.

To run locally:

Mind the `sample.env` file - by default the application will try to read AZD environment variables and falls back on `.env` only when it does not find one.

Activate the `.venv` virtual environment or run the binary directly:

```shell
cd src/backend
uv sync --prerelease=allow
source .venv/bin/activate
uvicorn app:app --port 8000
```

**Note**: The `--prerelease=allow` flag is required for the agent-framework package.

### Running the App locally - FRONTEND

The python project is managed by pyproject.toml and [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).
Install uv prior executing.

To run locally:

mind the sample.env file - by default the application will try to read AZD environment configuration and falls on .env only when it does not find one.

**Key Environment Variables for Agent Framework:**
```shell
PROJECT_ENDPOINT=https://your-foundry-project.services.ai.azure.com/api/projects/your-project
MODEL_DEPLOYMENT_NAME=gpt-4o-mini
HANDLER_TYPE=foundry_banking
APPLICATIONINSIGHTS_CONNECTION_STRING=your-connection-string
```

**OBS!** Activate .venv or run the binary directly.

```shell
cd src/frontend
uv sync
./.venv/bin/streamlit run app.py
```

### Usage

1. **Select Use Case**: Choose between `fsi_insurance`, `fsi_banking`, `energy` from the sidebar
2. **Start a Conversation**: Click "Start New Conversation" or select an existing one
3. **Chat**: Use the chat input to ask questions. Predefined questions are available in a dropdown
4. **Agents Online**: View the available agents for the selected use case
5. **Chat Histories**: View and reload your past conversations
