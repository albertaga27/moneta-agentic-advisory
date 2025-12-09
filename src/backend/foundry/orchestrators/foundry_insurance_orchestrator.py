"""
Moneta Insurance Orchestrator - Microsoft Agent Framework Implementation
Multi-agent orchestration using HandoffBuilder pattern for insurance services.

This orchestrator coordinates two specialized agents:
1. CRM Insurance Agent - Client insurance data and policy information
2. Policies Agent - Insurance policy research and product information

The orchestration uses the Handoff pattern where a coordinator agent routes
user requests to appropriate specialist agents.

Supports two modes:
1. Azure OpenAI mode (default) - Uses AzureOpenAIChatClient
2. Microsoft Foundry mode (--foundry) - Uses AIProjectClient with hosted agents
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Microsoft Agent Framework imports
from agent_framework import (
    ChatAgent,
    ChatMessage,
    HandoffBuilder,
    RequestInfoEvent,
    WorkflowOutputEvent,
    WorkflowEvent
)
from agent_framework._workflows._events import AgentRunEvent
from agent_framework.azure import AzureOpenAIChatClient, AzureAIAgentClient, AzureAIClient
from azure.identity.aio import AzureCliCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

# Import specialist agent functions from insurance agents subfolder
from foundry.agents.insurance.crm.crm_insurance_functions import crm_insurance_functions
from foundry.agents.insurance.policies.policies_functions import policies_functions

# Import agent management for Foundry mode
from foundry.agents.agent_management import AgentManager

# Setup Agent Framework observability - built-in OpenTelemetry integration
from agent_framework.observability import setup_observability, get_tracer, OtelAttr

# Load environment to get connection string
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# Configure Agent Framework observability
# The framework automatically traces: invoke_agent, chat, execute_tool spans
_app_insights_conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
_enable_sensitive_data = os.getenv("ENABLE_AI_CONTENT_RECORDING", "false").lower() == "true"
_otlp_endpoint = os.getenv("OTLP_ENDPOINT")

if _app_insights_conn_str:
    setup_observability(
        applicationinsights_connection_string=_app_insights_conn_str,
        enable_sensitive_data=_enable_sensitive_data
    )
    print("📊 Observability: Azure Application Insights")
elif _otlp_endpoint:
    setup_observability(
        otlp_endpoint=_otlp_endpoint,
        enable_sensitive_data=_enable_sensitive_data
    )
    print(f"📊 Observability: OTLP endpoint ({_otlp_endpoint})")
else:
    # Default to localhost for AI Toolkit
    setup_observability(
        otlp_endpoint="http://localhost:4317",
        enable_sensitive_data=_enable_sensitive_data
    )
    print("📊 Observability: Local OTLP (localhost:4317)")

# Get tracer for custom session-level spans
_tracer = get_tracer("moneta-insurance-orchestrator")


# Agent definitions for Foundry mode
AGENT_DEFINITIONS = {
    "coordinator": {
        "name": "moneta-insurance-coordinator",
        "instructions": (
            "You are the Moneta Insurance Coordinator. Analyze customer requests and route them to the appropriate specialist:\n"
            "- crm_insurance_agent: For client insurance data, policy information, client details, coverage summaries. "
            "Use when the request mentions a specific client name or ID and is about their policies.\n"
            "- policies_agent: For general insurance policy research, product information, coverage details, and policy recommendations.\n"
            "\n"
            "When you receive a request, immediately call the matching handoff tool "
            "(handoff_to_crm_insurance_agent or handoff_to_policies_agent) without explaining."
        ),
        "description": "Moneta Insurance Coordinator - routes requests to specialist agents"
    },
    "crm_insurance_agent": {
        "name": "moneta-crm-insurance-agent",
        "instructions": (
            "You are an Insurance CRM specialist. Help with client insurance data and policy information. "
            "Use your CRM functions to retrieve accurate customer policy data. "
            "ONLY use the provided functions - don't guess or use general knowledge. "
            "If client ID or name is not provided, politely inform the user that you need this information. "
            "Focus on policy details, coverage information, effective dates, and benefits."
        ),
        "description": "CRM Insurance Agent - handles client insurance data and policy information"
    },
    "policies_agent": {
        "name": "moneta-policies-agent",
        "instructions": (
            "You are an Insurance Policies specialist. Provide insurance policy information and product research insights. "
            "Use the search function to find relevant insurance policy information and product details. "
            "Provide CONCISE and actionable insurance insights focusing on coverage details, benefits, and recommendations."
        ),
        "description": "Policies Agent - provides insurance policy research and product information"
    }
}


class FoundryInsuranceOrchestrator:
    """
    Foundry Insurance Orchestrator compatible with Handler.py interface.
    
    This orchestrator uses Microsoft Agent Framework with HandoffBuilder pattern
    for multi-agent coordination in insurance services.
    """
    
    def __init__(self, use_foundry: bool = False):
        """
        Initialize the Foundry Insurance Orchestrator.
        
        Args:
            use_foundry: If True, uses Microsoft Foundry mode (hosted agents).
                        If False, uses Azure OpenAI mode (in-memory agents).
        """
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Foundry Insurance Orchestrator init")
        
        self.use_foundry = use_foundry
        self._workflow = None
        self._initialized = False
        
        # Configuration from environment
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.foundry_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("PROJECT_ENDPOINT")
        self.deployment_name = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT") or 
            os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
            os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME") or 
            os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
        )
        
        # Choose endpoint based on mode
        if self.use_foundry:
            self.endpoint = self.foundry_endpoint or self.openai_endpoint
        else:
            self.endpoint = self.openai_endpoint or self.foundry_endpoint
    
    async def _ensure_initialized(self):
        """Lazily initialize the workflow and agents."""
        if self._initialized:
            return
        
        # Validate Azure OpenAI endpoint is available
        if not self.openai_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required for the orchestrator workflow. "
                "The HandoffBuilder requires Azure OpenAI for proper tool calling support."
            )
        
        credential = AzureCliCredential()
        
        # Always use AzureOpenAIChatClient for the orchestrator workflow
        # because Foundry-hosted agents (AzureAIClient) don't properly support
        # tool calling with synthesized handoff tools.
        # 
        # The --foundry flag now primarily affects:
        # 1. Logging/tracing to indicate Foundry mode
        # 2. When combined with --new, agents are persisted to Foundry via AgentManager
        #
        # But the actual workflow always uses AzureOpenAIChatClient for reliable handoffs.
        self.logger.info(f"Using Azure OpenAI endpoint for workflow: {self.openai_endpoint}")
        chat_client = AzureOpenAIChatClient(
            endpoint=self.openai_endpoint,
            deployment_name=self.deployment_name,
            credential=credential
        )
        coordinator, crm_insurance_agent, policies_agent = create_specialist_agents(chat_client)
        
        # Build the handoff workflow with auto-registered handoff tools
        # auto_register_handoff_tools(True) synthesizes handoff_to_X tools for the coordinator
        self._workflow = (
            HandoffBuilder(
                name="moneta_insurance_handoff",
                participants=[coordinator, crm_insurance_agent, policies_agent],
            )
            .set_coordinator(coordinator)
            .auto_register_handoff_tools(True)  # This adds handoff_to_X tools to coordinator
            .with_termination_condition(
                lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 10
            )
            .build()
        )
        
        self._initialized = True
        self.logger.info(f"✅ Foundry Insurance Orchestrator initialized (foundry_mode={self.use_foundry}, using Azure OpenAI for handoffs)")
    
    async def process_conversation(self, user_id: str, conversation_messages: list, session_id: str = None) -> dict:
        """
        Process a conversation and return the agent's reply.
        
        This method is compatible with the Handler.py interface.
        
        Args:
            user_id: The user identifier
            conversation_messages: List of message dicts with 'role', 'name', 'content' keys
            session_id: Session/chat ID for tracing (typically the chat_id)
            
        Returns:
            Dict with 'role', 'name', 'content' keys representing the agent's reply
        """
        await self._ensure_initialized()
        
        # Use session_id for tracing, generate one if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Processing conversation with session_id: {session_id}")
        
        # Convert conversation history to ChatMessage objects for the workflow
        # This preserves the full context from CosmosDB across API calls
        chat_messages = []
        for msg in conversation_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            author_name = msg.get('name')  # Optional: agent name for assistant messages
            
            if content:  # Only add messages with content
                chat_msg = ChatMessage(
                    role=role,
                    text=content,
                    author_name=author_name
                )
                chat_messages.append(chat_msg)
        
        if not chat_messages:
            return {
                'role': 'assistant',
                'name': 'coordinator',
                'content': 'I didn\'t receive a message. How can I help you with insurance?'
            }
        
        self.logger.info(f"Injecting {len(chat_messages)} messages as conversation history")
        
        try:
            # Create a session-level tracing span
            from opentelemetry.trace import SpanKind
            
            with _tracer.start_as_current_span(
                "insurance_conversation",
                kind=SpanKind.SERVER,
                attributes={
                    str(OtelAttr.CONVERSATION_ID): session_id,
                    "session.id": session_id,
                    "user.id": user_id,
                    "session.mode": "foundry" if self.use_foundry else "azure_openai",
                    "conversation.message_count": len(chat_messages)
                }
            ) as session_span:
                # Run the workflow with the full conversation history
                # The workflow will use all messages as context
                final_response = ""
                responding_agent = "coordinator"
                
                async for event in self._workflow.run_stream(chat_messages):
                    # Check RequestInfoEvent which contains HandoffUserInputRequest with the conversation
                    # This is the primary way to get responses in the handoff pattern
                    if isinstance(event, RequestInfoEvent):
                        self.logger.debug(f"RequestInfoEvent: source={event.source_executor_id}, data type={type(event.data).__name__}")
                        if hasattr(event.data, 'conversation') and event.data.conversation:
                            # Get the last assistant message from the conversation
                            for msg in reversed(event.data.conversation):
                                if hasattr(msg, 'role') and msg.role.value == 'assistant':
                                    if hasattr(msg, 'text') and msg.text:
                                        final_response = msg.text
                                        responding_agent = getattr(msg, 'author_name', None) or getattr(event.data, 'awaiting_agent_id', 'coordinator')
                                        self.logger.info(f"Captured response from '{responding_agent}': {len(final_response)} chars")
                                        break
                    
                    # Also capture from AgentRunEvent as a backup
                    elif isinstance(event, AgentRunEvent):
                        if event.data and event.data.text:
                            final_response = event.data.text
                            responding_agent = event.executor_id or "coordinator"
                            self.logger.info(f"Captured from AgentRunEvent '{responding_agent}': {len(final_response)} chars")
                    
                    # Also check WorkflowOutputEvent for any additional output
                    elif isinstance(event, WorkflowOutputEvent):
                        if hasattr(event, 'data'):
                            if hasattr(event.data, 'text') and event.data.text:
                                final_response = event.data.text
                                responding_agent = getattr(event, 'source_executor_id', 'coordinator')
                                self.logger.info(f"Captured from WorkflowOutputEvent: {len(final_response)} chars")
                
                self.logger.info(f"Final response: {len(final_response)} chars from '{responding_agent}'")
                
                # Record response info in span
                session_span.set_attribute("response.agent", responding_agent)
                session_span.set_attribute("response.length", len(final_response))
                
                return {
                    'role': 'assistant',
                    'name': responding_agent,
                    'content': final_response or 'I apologize, but I was unable to generate a response.'
                }
            
        except Exception as e:
            self.logger.error(f"Error processing conversation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'role': 'assistant',
                'name': 'coordinator',
                'content': f'I encountered an error while processing your request. Please try again.'
            }


def create_specialist_agents(chat_client: AzureOpenAIChatClient) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """
    Create the coordinator and specialist agents for the insurance workflow.
    Uses Azure OpenAI mode with in-memory agents.
    
    Args:
        chat_client: The Azure OpenAI chat client
        
    Returns:
        Tuple of (coordinator, crm_insurance_agent, policies_agent)
    """
    
    # Coordinator agent - routes requests to specialists
    coordinator = chat_client.create_agent(
        instructions=AGENT_DEFINITIONS["coordinator"]["instructions"],
        name="coordinator"
    )
    
    # CRM Insurance Agent - handles client insurance data
    crm_insurance_agent = chat_client.create_agent(
        instructions=AGENT_DEFINITIONS["crm_insurance_agent"]["instructions"],
        name="crm_insurance_agent",
        tools=crm_insurance_functions
    )
    
    # Policies Agent - handles insurance policy research
    policies_agent = chat_client.create_agent(
        instructions=AGENT_DEFINITIONS["policies_agent"]["instructions"],
        name="policies_agent",
        tools=policies_functions
    )
    
    print(f"✅ Created coordinator agent: {coordinator.name}")
    print(f"✅ Created CRM Insurance agent: {crm_insurance_agent.name}")
    print(f"✅ Created Policies agent: {policies_agent.name}")
    
    return coordinator, crm_insurance_agent, policies_agent


async def get_existing_agent_ids(
    project_endpoint: str,
    credential: AzureCliCredential
) -> dict[str, str]:
    """
    Get IDs of existing agents in the Foundry project.
    
    Returns:
        Dictionary mapping agent names to their IDs (most recent version)
    """
    from azure.ai.agents.aio import AgentsClient
    
    agent_ids = {}
    async with AgentsClient(endpoint=project_endpoint, credential=credential) as client:
        # list_agents returns an async iterator
        async for agent in client.list_agents():
            if agent.name and agent.id:
                # Keep the first (most recent) agent with each name
                if agent.name not in agent_ids:
                    agent_ids[agent.name] = agent.id
    
    return agent_ids


async def create_persistent_foundry_agents(
    project_endpoint: str,
    credential: AzureCliCredential,
    model_deployment_name: str
) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """
    Create persistent agents in Microsoft Foundry using AgentManager, then wrap them
    with ChatAgent for use in the orchestrator workflow.
    
    This creates actual Foundry-hosted agents that are visible in the Foundry UI
    and persist across sessions.
    
    Note: Tools (Python functions) are NOT registered with Foundry - they are bound
    to the local ChatAgent wrapper. Foundry stores the agent definition (name, 
    instructions, model) while local tools execute at runtime.
    
    Args:
        project_endpoint: The Azure AI Project endpoint URL
        credential: Azure credential for authentication
        model_deployment_name: The model deployment name to use
        
    Returns:
        Tuple of (coordinator, crm_insurance_agent, policies_agent)
    """
    print(f"\n🔧 Creating persistent Foundry agents...")
    print(f"   Agents will be visible in Foundry UI")
    print(f"   Tools are bound locally (not registered with Foundry)")
    print()
    
    agents = []
    
    # Use AgentManager to create persistent agents in Foundry
    async with AgentManager() as manager:
        for agent_key in ["coordinator", "crm_insurance_agent", "policies_agent"]:
            agent_def = AGENT_DEFINITIONS[agent_key]
            
            # Get tools for this agent (used locally, not sent to Foundry)
            tools = None
            if agent_key == "crm_insurance_agent":
                tools = crm_insurance_functions
            elif agent_key == "policies_agent":
                tools = policies_functions
            
            agent_name = agent_def["name"]
            
            # Create persistent agent in Foundry via AgentManager
            # Note: We don't pass tools here - Foundry stores the definition only
            foundry_agent = await manager.create_agent(
                agent_name=agent_name,
                instructions=agent_def["instructions"],
                model=model_deployment_name
                # tools are NOT passed - they're local Python functions
            )
            
            print(f"✅ Created in Foundry: {foundry_agent['name']} (ID: {foundry_agent['id']})")
            
            # Now create a ChatAgent wrapper using AzureAIClient to use this Foundry agent
            client = AzureAIClient(
                project_endpoint=project_endpoint,
                model_deployment_name=model_deployment_name,
                async_credential=credential,
                agent_name=agent_name,
                use_latest_version=True,  # Use the version we just created
                should_cleanup_agent=False
            )
            
            # Create ChatAgent that references the Foundry agent
            # Tools are bound here for local execution
            agent = client.create_agent(
                name=agent_key,
                instructions=agent_def["instructions"],
                tools=tools
            )
            
            agents.append(agent)
    
    print(f"\n✅ All {len(agents)} agents created and registered in Foundry")
    return tuple(agents)


async def create_foundry_agents(
    project_endpoint: str,
    credential: AzureCliCredential,
    model_deployment_name: str,
    use_existing: bool = True,
    agent_version: str | None = None,
    use_latest_version: bool = True,
    force_new_version: bool = False
) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """
    Create or reuse agents using AzureAIClient (v2 API) for Microsoft Foundry project.
    
    This uses the new AzureAIClient which supports agent versioning:
    - agent_name: Name to use when creating/finding agents in Foundry
    - agent_version: Specific version to use (e.g., "1.0", "2.0")
    - use_latest_version: If True, uses the latest existing version if available
    
    Note: This creates in-memory agent wrappers that reference Foundry agents.
    To create persistent agents visible in Foundry UI, use --new flag with --foundry.
    
    Args:
        project_endpoint: The Azure AI Project endpoint URL
        credential: Azure credential for authentication
        model_deployment_name: The model deployment name to use
        use_existing: If True, reuse existing agents; otherwise create new ones
        agent_version: Specific agent version to use (e.g., "1.0")
        use_latest_version: If True, use latest existing version in Foundry
        force_new_version: If True, create new version even if agent exists
        
    Returns:
        Tuple of (coordinator, crm_insurance_agent, policies_agent)
    """
    agents = []
    
    # Log versioning configuration
    print(f"\n🔧 Agent Configuration:")
    print(f"   use_existing: {use_existing}")
    print(f"   agent_version: {agent_version or 'auto'}")
    print(f"   use_latest_version: {use_latest_version}")
    print(f"   force_new_version: {force_new_version}")
    print()
    
    for agent_key in ["coordinator", "crm_insurance_agent", "policies_agent"]:
        agent_def = AGENT_DEFINITIONS[agent_key]
        tools = None
        if agent_key == "crm_insurance_agent":
            tools = crm_insurance_functions
        elif agent_key == "policies_agent":
            tools = policies_functions
        
        agent_name = agent_def["name"]
        
        # Build AzureAIClient with versioning support
        # The v2 API automatically handles agent reuse via agent_name
        client = AzureAIClient(
            project_endpoint=project_endpoint,
            model_deployment_name=model_deployment_name,
            async_credential=credential,
            agent_name=agent_name,
            agent_version=agent_version if not use_latest_version else None,
            use_latest_version=use_latest_version and not force_new_version,
            should_cleanup_agent=False
        )
        
        # Create agent - the client handles reuse logic based on agent_name + version
        agent = client.create_agent(
            name=agent_key,
            instructions=agent_def["instructions"],
            tools=tools
        )
        
        version_info = f"v{agent_version}" if agent_version else ("latest" if use_latest_version else "new")
        print(f"✅ Agent ready: {agent_name} ({version_info})")
        
        agents.append(agent)
    
    return tuple(agents)


async def handle_workflow_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """
    Process workflow events and return any pending input requests.
    The Agent Framework automatically traces agent invocations and handoffs.
    
    Args:
        events: List of workflow events to process
        
    Returns:
        List of pending input requests
    """
    pending_requests = []
    
    for event in events:
        if isinstance(event, RequestInfoEvent):
            pending_requests.append(event)
            request_data = event.data
            awaiting_agent = request_data.awaiting_agent_id.upper()
            
            print(f"\n{'='*60}")
            print(f"AWAITING INPUT FROM: {awaiting_agent}")
            print(f"{'='*60}")
            
            # Show recent conversation context (full history maintained by framework)
            for msg in request_data.conversation[-3:]:
                author = msg.author_name or msg.role.value
                print(f"    [{author}]: {msg.text}")
                
        elif isinstance(event, WorkflowOutputEvent):
            print("\n✅ Workflow completed!")
            
    return pending_requests


async def main():
    """
    Main function to initialize and run the Moneta Insurance orchestrator.
    This orchestrator coordinates CRM Insurance and Policies agents using the Handoff pattern.
    
    Supports two modes:
    - Azure OpenAI mode (default): Uses in-memory agents with AzureOpenAIChatClient
    - Foundry mode (--foundry): Uses Foundry-hosted agents with AzureAIClient (v2 API)
    
    Foundry mode supports agent versioning:
    - --version X.Y: Use specific agent version
    - --latest: Use latest existing version (default)
    - --force-new: Create new version even if agents exist
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Moneta Insurance Multi-Agent Orchestrator")
    parser.add_argument("--foundry", action="store_true", help="Use Microsoft Foundry mode (hosted agents)")
    parser.add_argument("--new", action="store_true", help="Create new persistent agents in Foundry (requires --foundry)")
    parser.add_argument("--version", type=str, default=None, help="Specific agent version to use (e.g., 1.0)")
    parser.add_argument("--latest", action="store_true", default=True, help="Use latest existing version (default)")
    parser.add_argument("--force-new-version", action="store_true", help="Create new version even if agents exist")
    
    args = parser.parse_args()
    
    use_foundry = args.foundry
    use_existing = not args.new
    agent_version = args.version
    use_latest_version = args.latest and agent_version is None  # Only use latest if no specific version
    force_new_version = args.force_new_version
    
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
    
    # Load environment variables from backend directory (where .env is located)
    # Path: orchestrators -> foundry -> backend/.env
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    
    # Configuration - support multiple environment variable names
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    foundry_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("PROJECT_ENDPOINT")
    deployment_name = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT") or 
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
        os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME") or 
        os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
    )
    
    # Choose endpoint based on mode
    if use_foundry:
        endpoint = foundry_endpoint or openai_endpoint
        print("🏗️  Using Microsoft Foundry mode (hosted agents with AzureAIClient v2)")
    else:
        endpoint = openai_endpoint or foundry_endpoint
        print("🔌 Using Azure OpenAI mode (in-memory agents)")
    
    # Validate required environment variables
    if not endpoint:
        print("❌ Error: No endpoint configured")
        print("Set AZURE_OPENAI_ENDPOINT for Azure OpenAI mode")
        print("Set AZURE_AI_PROJECT_ENDPOINT for Foundry mode")
        return
    
    try:
        # Create Azure CLI credential (async version)
        async with AzureCliCredential() as credential:
            
            if use_foundry:
                # Foundry mode - use AzureAIClient (v2 API) with hosted agents
                if not use_existing:
                    # --new flag: Create persistent agents in Foundry (visible in UI)
                    print("📝 Creating new persistent agents in Foundry...")
                    coordinator, crm_insurance_agent, policies_agent = await create_persistent_foundry_agents(
                        project_endpoint=endpoint,
                        credential=credential,
                        model_deployment_name=deployment_name
                    )
                else:
                    # Reuse existing Foundry-hosted agents with version support
                    coordinator, crm_insurance_agent, policies_agent = await create_foundry_agents(
                        project_endpoint=endpoint,
                        credential=credential,
                        model_deployment_name=deployment_name,
                        use_existing=use_existing,
                        agent_version=agent_version,
                        use_latest_version=use_latest_version,
                        force_new_version=force_new_version
                    )
                
                # Run the workflow
                await run_workflow(
                    coordinator, crm_insurance_agent, policies_agent,
                    use_foundry
                )
            else:
                # Azure OpenAI mode - use AzureOpenAIChatClient with in-memory agents
                chat_client = AzureOpenAIChatClient(
                    endpoint=endpoint,
                    deployment_name=deployment_name,
                    credential=credential
                )
                
                # Create all agents
                coordinator, crm_insurance_agent, policies_agent = create_specialist_agents(chat_client)
                
                # Run the workflow
                await run_workflow(
                    coordinator, crm_insurance_agent, policies_agent,
                    use_foundry
                )
    
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {str(e)}")
        print("Please check your environment variables and Azure credentials.")
        import traceback
        traceback.print_exc()


async def run_workflow(
    coordinator: ChatAgent,
    crm_insurance_agent: ChatAgent,
    policies_agent: ChatAgent,
    use_foundry: bool
):
    """
    Run the multi-agent workflow with the given agents.
    
    Tracing is handled at two levels:
    1. Agent Framework built-in: invoke_agent, chat, execute_tool spans
    2. Custom session-level: conversation turns, session metrics
    
    The HandoffWorkflow maintains full conversation history internally.
    """
    import uuid
    from opentelemetry.trace import SpanKind
    
    # Generate unique session ID for this conversation
    session_id = str(uuid.uuid4())[:8]
    
    # Note: We use session_id in OpenTelemetry span attributes for trace correlation
    # Do NOT set chat_options.conversation_id - that field is used by Azure AI API
    # for response chaining (previous_response_id) and expects 'resp...' format
    
    # Build the handoff workflow using HandoffBuilder
    # The framework maintains full conversation history across all agent interactions
    # auto_register_handoff_tools(True) synthesizes handoff_to_X tools for the coordinator
    workflow = (
        HandoffBuilder(
            name="moneta_insurance_handoff",
            participants=[coordinator, crm_insurance_agent, policies_agent],
        )
        .set_coordinator(coordinator)
        .auto_register_handoff_tools(True)  # This adds handoff_to_X tools to coordinator
        .with_termination_condition(
            lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 10
        )
        .build()
    )
    
    mode_label = "Microsoft Foundry" if use_foundry else "Azure OpenAI"
    print("\n" + "="*60)
    print("🛡️ MONETA INSURANCE MULTI-AGENT ORCHESTRATOR")
    print(f"   Mode: {mode_label}")
    print(f"   Session ID: {session_id}  (for trace correlation)")
    print("="*60)
    print("\n🎭 Orchestrator with Specialist Agents:")
    print("   • Coordinator - Routes your requests")
    print("   • CRM Insurance Agent - Client insurance data & policies")
    print("   • Policies Agent - Insurance policy research & products")
    print("\nType 'exit' or 'quit' to end the conversation.")
    print("="*60 + "\n")
    
    # Interactive conversation loop with session-level tracing
    turn_counter = 0
    
    # Start a session-level span that wraps the entire conversation
    # Use gen_ai.conversation.id for Foundry trace correlation
    with _tracer.start_as_current_span(
        "insurance_session",
        kind=SpanKind.SERVER,
        attributes={
            str(OtelAttr.CONVERSATION_ID): session_id,  # Foundry trace correlation
            "session.id": session_id,
            "session.mode": "foundry" if use_foundry else "azure_openai"
        }
    ) as session_span:
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                # Record session summary
                session_span.set_attribute("session.total_turns", turn_counter)
                session_span.set_attribute("session.status", "completed")
                print("\n👋 Thank you for using Moneta Insurance Services!")
                break
            
            if not user_input:
                print("Please enter a valid query.")
                continue
            
            turn_counter += 1
            
            try:
                # Create a turn-level span (child of session span)
                # Agent Framework will create child spans for invoke_agent, chat, execute_tool
                with _tracer.start_as_current_span(
                    f"conversation_turn_{turn_counter}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        str(OtelAttr.CONVERSATION_ID): session_id,  # Foundry trace correlation
                        "turn.number": turn_counter,
                        "turn.user_input": user_input[:500],
                        "session.id": session_id
                    }
                ) as turn_span:
                    handoffs = []
                    
                    # Run workflow - framework traces all agent interactions automatically
                    events = [event async for event in workflow.run_stream(user_input)]
                    
                    # Track which agents were involved
                    for event in events:
                        if isinstance(event, RequestInfoEvent):
                            if hasattr(event, 'data') and hasattr(event.data, 'awaiting_agent_id'):
                                handoffs.append(event.data.awaiting_agent_id)
                    
                    pending_requests = await handle_workflow_events(events)
                    
                    # Record handoff chain for this turn
                    if handoffs:
                        turn_span.set_attribute("turn.handoffs", ",".join(handoffs))
                        turn_span.set_attribute("turn.handoff_count", len(handoffs))
                
                # Handle follow-up requests if any
                while pending_requests:
                    user_response = input("\nYou: ").strip()
                    if not user_response:
                        break
                    
                    with _tracer.start_as_current_span(
                        f"followup_turn_{turn_counter}",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            str(OtelAttr.CONVERSATION_ID): session_id,  # Foundry trace correlation
                            "turn.number": turn_counter,
                            "turn.is_followup": True,
                            "turn.user_input": user_response[:500],
                            "session.id": session_id
                        }
                    ) as followup_span:
                        responses = {req.request_id: user_response for req in pending_requests}
                        events = [event async for event in workflow.send_responses_streaming(responses)]
                        pending_requests = await handle_workflow_events(events)
                
                print()  # Add blank line after response
                
            except Exception as e:
                print(f"❌ Error processing request: {str(e)}")
                # Record error in session span
                session_span.set_attribute("session.has_errors", True)
                session_span.record_exception(e)
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
