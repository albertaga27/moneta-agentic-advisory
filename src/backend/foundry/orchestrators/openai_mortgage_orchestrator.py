"""
Mortgage Orchestrator - Sequential Workflow with Human-in-the-Loop

This orchestrator uses Microsoft Agent Framework's SequentialBuilder to coordinate
mortgage application processing through three specialized agents:

1. Classifier Agent - Classifies the application type, completeness, and risk
2. Document Extraction Agent - Extracts data from submitted documents
3. Policy Check Agent - Verifies compliance with regulatory requirements

The workflow includes Human-in-the-Loop (HIL) support, allowing human reviewers
to provide feedback after the Document Extraction agent completes, ensuring
extracted data is accurate before proceeding to policy checks.

Usage:
    orchestrator = MortgageOrchestrator()
    await orchestrator.initialize()
    result = await orchestrator.process_mortgage_request(mortgage_request)
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import base64
from dataclasses import asdict
from datetime import datetime as dt
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field
from fastapi import HTTPException

# Microsoft Agent Framework imports
from agent_framework import (
    ChatAgent,
    ChatMessage,
    Role,
    SequentialBuilder,
    HandoffBuilder,
    RequestInfoEvent,
    WorkflowOutputEvent,
    WorkflowStatusEvent,
    WorkflowRunState,
    ExecutorCompletedEvent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential

# Import mortgage components from banking agents
from foundry.agents.banking.mortgage.models import MortgageRequest
from foundry.agents.banking.mortgage.classifier_functions import classifier_functions
from foundry.agents.banking.mortgage.document_extraction_functions import document_extraction_functions
from foundry.agents.banking.mortgage.policy_check_functions import policy_check_functions
from foundry.agents.banking.mortgage.query_functions import (
    query_functions,
    set_query_context,
    clear_query_context as clear_query_func_context
)

# Add backend root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tracing import setup_tracing, get_tracer, set_conversation_context, clear_conversation_context
from mortgage_store import CosmosPersister

# Load environment
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# Setup tracing
setup_tracing()
_tracer = get_tracer("moneta-mortgage-orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models for API
# ─────────────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    """Document with filename and base64 content."""
    filename: str
    b64: str = Field(..., description="Base-64 content of the file")


class RequestData(BaseModel):
    """Typed request data for mortgage application."""
    property_value: float              # CHF
    income: float                      # yearly net CHF
    liabilities: float                 # yearly CHF
    down_payment_pc: float             # e.g. 20.0 for 20%
    swiss_or_eu_citizen: bool = True  # Lex Koller relevance
    # Pillar pledges
    pledge_p2: bool = False
    pledge_p2_amount: float = Field(0.0, ge=0)
    pledge_p3: bool = False
    pledge_p3_amount: float = Field(0.0, ge=0)


class RequestPayload(BaseModel):
    """Payload for creating a new mortgage request."""
    user_id: str
    request_id: str | None = Field(
        None,
        description="Optional GUID created client-side; server will generate if omitted",
    )
    request_data: RequestData
    documents: Dict[str, str]  # {"salary.pdf": "https://…", …}


# Agent instructions
CLASSIFIER_INSTRUCTIONS = """You are a file-routing assistant for Moneta Bank's Mortgage Processing System.

Based only on the filenames provided, map each document to one of the following categories:
- identity (ID, driver license or passport)
- residence_permit (if non-Swiss)
- proof_of_income (salary slips, tax return, accounts statements)
- credit_history (Betreibungsauszug)
- property_valuation (listing, appraisal)
- lex_koller_authorisation (Lex Koller authorization for foreign buyers)
- land_registry_extract (Grundbuchauszug)
- building_insurance (building insurance certificate or policy)
- purchase_contract (draft purchase contract or reservation agreement)
- other (unclassified documents)

Use the 'classify_documents' function with the document list.
Reply with valid JSON: {"classification": {<filename>: <label>, ...}}
Be CONCISE - only provide the classification results."""

DOCUMENT_EXTRACTION_INSTRUCTIONS = """You are a Document Extraction Specialist for Moneta Bank.

Extract data from the mortgage documents. For each document:
1. Use 'extract_document_data' function with the document filename and its classified type
2. Collect the extracted fields from each document
3. Return results as JSON with extracted data keyed by filename

**Output Format:**
Return a JSON object with extracted data for each document:
{
  "document_name.pdf": {
    "field1": {"value": "...", "confidence": 0.95},
    "field2": {"value": "...", "confidence": 0.88}
  },
  ...
}

Flag any documents with low confidence scores (<0.80) for review."""

POLICY_CHECK_INSTRUCTIONS = """You are a Regulatory Compliance Officer for Moneta Bank in Switzerland.

Verify the mortgage application against Swiss regulatory requirements:
1. Use 'check_policy_compliance' with the full application data including classification results
2. The application data should include: data (form fields), documents, and classification
3. Provide clear decision: approved=true/false with rule_checks array

Swiss mortgage rules checked:
- Down-payment ≥ 20%, Hard-equity ≥ 10%, LTV ≤ 80%, Affordability ≤ 33%
- Amortisation to ≤ 66.7% in 15 years, Lex Koller, Cross-border, Document completeness

Return the complete policy check result with rule_checks array."""


# Router Agent Instructions - determines if new or existing request
ROUTER_INSTRUCTIONS = """You are the Moneta Bank Mortgage Request Router. Your job is to analyze user messages 
and determine if they want to:

1. **NEW_REQUEST**: Create a new mortgage application or submit a new mortgage request
   - Keywords: "new mortgage", "apply for mortgage", "start application", "submit mortgage", "new application"
   
2. **EXISTING_REQUEST**: Operate on an existing mortgage request
   - Keywords: "my request", "check status", "update documents", "existing application", "request ID", "MR-"
   - User mentions a specific request ID or references an ongoing application

IMPORTANT: Analyze the user message carefully and respond with ONLY one of:
- "NEW_REQUEST" if user wants to create a new mortgage application
- "EXISTING_REQUEST" if user wants to operate on an existing mortgage request
- "EXISTING_REQUEST" if the context contains mortgage request details (status, documents, classification results)

Do NOT provide any other text - just the category."""


# Coordinator Agent Instructions - handles existing requests with handoff
MORTGAGE_COORDINATOR_INSTRUCTIONS = """You are the Moneta Bank Mortgage Coordinator for existing applications.

You help customers with their ongoing mortgage applications. You have query tools to find requests and can route to specialists.

**Your Query Tools:**
- list_mortgage_requests: List all requests (with optional status filter)
- get_request_by_id: Get details for a specific request ID
- search_requests: Search requests by any term (property value, dates, etc.)
- get_latest_request: Get the user's most recent request
- get_requests_by_status: Filter requests by status (PENDING_REVIEW, APPROVED, etc.)

**When to use query tools:**
- User asks "show my requests" or "list my applications" → use list_mortgage_requests
- User mentions a request ID like "MR-2024-001" → use get_request_by_id
- User says "my request" without ID → use get_latest_request
- User asks about "pending" or "approved" requests → use get_requests_by_status
- User gives partial info → use search_requests

**Specialist Agents (handoff when needed):**
- **classifier**: For document classification questions, re-classification, or missing document types
- **document_extractor**: For document extraction status, re-extraction, or data verification
- **policy_checker**: For compliance status, policy check results, or approval requirements

**Workflow:**
1. First, use your query tools to find the relevant request(s)
2. Provide the user with request information
3. If they need specialist help, use handoff tools (handoff_to_classifier, handoff_to_document_extractor, handoff_to_policy_checker)

Always be helpful and provide clear information about the request status."""


# Specialist agents for handoff workflow (existing requests)
CLASSIFIER_HANDOFF_INSTRUCTIONS = """You are a Document Classification Specialist for Moneta Bank.

You help with existing mortgage applications by:
1. Reviewing document classifications already performed
2. Re-classifying documents if requested
3. Identifying missing document types
4. Explaining document categorization

Use the 'classify_documents' function when classification operations are needed.
Always reference the current request context when helping the user."""

EXTRACTOR_HANDOFF_INSTRUCTIONS = """You are a Document Extraction Specialist for Moneta Bank.

You help with existing mortgage applications by:
1. Reviewing extracted data from documents
2. Re-extracting data if corrections are needed
3. Verifying extraction accuracy
4. Explaining what data was extracted and confidence levels

Use 'extract_document_data' function when extraction operations are needed.
Always reference the current request context when helping the user."""

POLICY_HANDOFF_INSTRUCTIONS = """You are a Policy Compliance Specialist for Moneta Bank.

You help with existing mortgage applications by:
1. Reviewing compliance check results
2. Explaining what policy checks passed or failed
3. Providing guidance on what's needed for approval
4. Running additional policy checks if data has been updated

Use 'check_policy_compliance' function when compliance checks are needed.
Always reference the current request context when helping the user."""


class MortgageOrchestrator:
    """
    Mortgage Orchestrator with Sequential workflow and Human-in-the-Loop support.
    
    Coordinates mortgage application processing through specialized agents
    with optional human review after document extraction.
    
    Supports two workflows:
    1. Sequential workflow (new requests): Classifier -> Document Extractor -> Policy Checker
    2. Handoff workflow (existing requests): Coordinator routes to specialist agents as needed
    """
    
    def __init__(
        self,
        enable_human_review: bool = True,
        human_review_agents: list[str] | None = None,
        persister: CosmosPersister | None = None,
    ):
        """
        Initialize the Mortgage Orchestrator.
        
        Args:
            enable_human_review: Enable Human-in-the-Loop for specified agents
            human_review_agents: List of agent names to pause for human review.
                               Defaults to ["document_extractor"] if enable_human_review is True.
            persister: Optional CosmosPersister instance. If None, creates one from env vars.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Mortgage Orchestrator")
        
        self._sequential_workflow = None  # For new requests
        self._handoff_workflow = None     # For existing requests
        self._router_agent = None         # For routing decisions
        self._initialized = False
        self._chat_client = None
        self._credential = None
        
        self.enable_human_review = enable_human_review
        self.human_review_agents = human_review_agents or ["document_extractor"]
        
        # Initialize persistence layer
        self._persister = persister or CosmosPersister()
        
        # Configuration from environment
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and build both workflows."""
        if self._initialized:
            return
        
        self.logger.info("Building mortgage processing workflows...")
        
        # Initialize credential and client
        self._credential = AzureCliCredential()
        self._chat_client = AzureOpenAIChatClient(
            endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            credential=self._credential
        )
        
        # Create router agent for determining workflow type
        self._router_agent = self._chat_client.as_agent(
            name="router",
            instructions=ROUTER_INSTRUCTIONS
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Sequential Workflow (for new mortgage requests)
        # ─────────────────────────────────────────────────────────────────────
        classifier_agent = self._chat_client.as_agent(
            name="classifier",
            instructions=CLASSIFIER_INSTRUCTIONS,
            tools=classifier_functions
        )
        
        document_extractor_agent = self._chat_client.as_agent(
            name="document_extractor",
            instructions=DOCUMENT_EXTRACTION_INSTRUCTIONS,
            tools=document_extraction_functions
        )
        
        policy_checker_agent = self._chat_client.as_agent(
            name="policy_checker",
            instructions=POLICY_CHECK_INSTRUCTIONS,
            tools=policy_check_functions
        )
        
        # Build sequential workflow with Human-in-the-Loop
        builder = (
            SequentialBuilder()
            .participants([classifier_agent, document_extractor_agent, policy_checker_agent])
        )
        
        # Enable human review for specified agents
        if self.enable_human_review:
            builder = builder.with_request_info(agents=self.human_review_agents)
        
        self._sequential_workflow = builder.build()
        
        # ─────────────────────────────────────────────────────────────────────
        # Handoff Workflow (for existing mortgage requests)
        # ─────────────────────────────────────────────────────────────────────
        coordinator_agent = self._chat_client.as_agent(
            name="mortgage-coordinator",
            instructions=MORTGAGE_COORDINATOR_INSTRUCTIONS,
            tools=query_functions  # Query tools for finding/filtering requests
        )
        
        classifier_handoff_agent = self._chat_client.as_agent(
            name="classifier",
            instructions=CLASSIFIER_HANDOFF_INSTRUCTIONS,
            tools=classifier_functions
        )
        
        extractor_handoff_agent = self._chat_client.as_agent(
            name="document_extractor",
            instructions=EXTRACTOR_HANDOFF_INSTRUCTIONS,
            tools=document_extraction_functions
        )
        
        policy_handoff_agent = self._chat_client.as_agent(
            name="policy_checker",
            instructions=POLICY_HANDOFF_INSTRUCTIONS,
            tools=policy_check_functions
        )
        
        # Build handoff workflow for existing requests
        self._handoff_workflow = (
            HandoffBuilder(
                name="mortgage_existing_request_handoff",
                participants=[
                    coordinator_agent,
                    classifier_handoff_agent,
                    extractor_handoff_agent,
                    policy_handoff_agent
                ]
            )
            .with_start_agent(coordinator_agent)
            .with_termination_condition(
                lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 10
            )
            .build()
        )
        
        self._initialized = True
        
        self.logger.info(
            f"✅ Mortgage workflows initialized. "
            f"Sequential (new requests): 3 agents. "
            f"Handoff (existing requests): 4 agents. "
            f"Human review enabled: {self.enable_human_review}"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Persistence Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _load_user_doc(self, user_id: str) -> Dict[str, Any]:
        """Load user document from Cosmos DB."""
        return self._persister.load_user_doc(user_id)
    
    def _save_user_doc(self, user_doc: Dict[str, Any]) -> None:
        """Save user document to Cosmos DB."""
        self._persister.save_user_doc(user_doc)
    
    def create_request(
        self,
        user_id: str,
        form_data: dict,
        documents: dict[str, str],
        request_id: str | None = None,
    ) -> str:
        """
        Create a new mortgage request for a user.
        
        Args:
            user_id: The user ID
            form_data: The mortgage application form data
            documents: Dictionary mapping filenames to URLs (or base64 data URIs)
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            The request ID
        """
        request_id = request_id or str(uuid.uuid4())
        self.logger.info(f"Creating new request {request_id}, for user_id={user_id}...")

        # For persistence, only store document filenames, not base64 content
        # This keeps CosmosDB documents small and avoids storing binary data
        document_filenames = {}
        for filename, content in documents.items():
            if content.startswith('data:'):
                # It's a base64 data URI - just store the filename
                document_filenames[filename] = f"file://{filename}"
            else:
                # It's already a URL/path
                document_filenames[filename] = content

        new_req = MortgageRequest(
            request_id=request_id,
            data=form_data,
            documents=document_filenames,
        )

        user_doc = self._load_user_doc(user_id)
        self.logger.info(f"Appending new request {request_id}, to user_id={user_id} structure...")
        user_doc["requests"].append(asdict(new_req))  # append once
        self._save_user_doc(user_doc)                  # persist once
        return request_id
    
    def get_request_list(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Return a lightweight list of requests for the given user.
        
        Shape: [{request_id, created_utc, status, property_value, swiss_or_eu_citizen, decision}]
        """
        doc = self._load_user_doc(user_id)
        return [
            {
                "request_id": r["request_id"],
                "created_utc": r.get("created_utc"),
                "status": r.get("status"),
                "property_value": r.get("data", {}).get("property_value"),
                "swiss_or_eu_citizen": r.get("data", {}).get("swiss_or_eu_citizen"),
                "decision": r.get("decision"),
            }
            for r in doc.get("requests", [])
        ]
    
    def get_request_detail(self, user_id: str, request_id: str) -> Dict[str, Any]:
        """
        Return the full request dict or raise 404 if not found.
        """
        doc = self._load_user_doc(user_id)
        for r in doc.get("requests", []):
            if r["request_id"] == request_id:
                return r
        raise HTTPException(status_code=404, detail="Request not found")
    
    def update_request(
        self,
        user_id: str,
        request_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update an existing request with new data.
        
        Args:
            user_id: The user ID
            request_id: The request ID to update
            updates: Dictionary of fields to update
            
        Returns:
            The updated request dict
        """
        user_doc = self._load_user_doc(user_id)
        request_ref = self._persister.get_request_ref(user_doc, request_id)
        
        # Update fields
        for key, value in updates.items():
            request_ref[key] = value
        request_ref["updated_utc"] = dt.utcnow().isoformat()
        
        self._save_user_doc(user_doc)
        return request_ref
    
    def _parse_agent_json_response(self, response_text: str) -> Dict[str, Any] | None:
        """
        Parse JSON from agent response text.
        
        Agents may include JSON within markdown code blocks or plain text.
        This method extracts and parses the JSON content.
        """
        import re
        
        if not response_text:
            return None
        
        # Try to find JSON in code blocks first
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response_text)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find the outermost JSON object by finding balanced braces
        # Start from the first { and find its matching }
        start_idx = response_text.find('{')
        if start_idx == -1:
            return None
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(response_text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count == 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _format_rule_checks_for_display(self, rule_checks: List[Dict[str, Any]]) -> str:
        """Format rule checks as a nicely readable string."""
        lines = []
        for rc in rule_checks:
            status = rc.get('status', 'unknown')
            rule = rc.get('rule', 'Unknown rule')
            details = rc.get('details', '')
            
            if status == 'pass':
                icon = '✅'
            elif status == 'warning':
                icon = '⚠️'
            else:
                icon = '❌'
            
            lines.append(f"{icon} **{rule}**: {status.upper()}")
            
            # Format details
            if isinstance(details, dict):
                for key, value in details.items():
                    status_icon = '✓' if value == 'present' else '✗'
                    lines.append(f"   - {key}: {status_icon} {value}")
            elif details:
                lines.append(f"   {details}")
        
        return '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Conversation Processing (called from handler.py)
    # ─────────────────────────────────────────────────────────────────────────
    
    async def process_conversation(
        self, 
        user_id: str, 
        conversation_messages: list, 
        session_id: str = None
    ) -> dict:
        """
        Process a conversation and return the agent's reply.
        
        This is the main entry point called from handler.py. It routes the request
        to either the sequential workflow (new mortgage requests) or the handoff
        workflow (existing mortgage requests) based on the user's intent.
        
        Args:
            user_id: The user identifier
            conversation_messages: List of message dicts with 'role', 'name', 'content' keys
            session_id: Session/chat ID for tracing (typically the chat_id)
            
        Returns:
            Dict with 'role', 'name', 'content' keys representing the agent's reply
        """
        await self.initialize()
        
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Processing mortgage conversation with session_id: {session_id}")
        
        # Get the latest user message
        latest_message = ""
        for msg in reversed(conversation_messages):
            if msg.get('role') == 'user':
                latest_message = msg.get('content', '')
                break
        
        if not latest_message:
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': "I didn't receive a message. How can I help you with your mortgage application?"
            }
        
        try:
            from opentelemetry.trace import SpanKind
            
            set_conversation_context(
                conversation_id=session_id,
                user_id=user_id,
                mode="azure_openai"
            )
            
            with _tracer.start_as_current_span(
                "mortgage_conversation",
                kind=SpanKind.SERVER,
                attributes={
                    "gen_ai.conversation.id": session_id,
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.provider.name": "azure_openai",
                    "gen_ai.agent.name": "mortgage_orchestrator",
                    "session.id": session_id,
                    "user.id": user_id,
                    "session.mode": "azure_openai",
                    "conversation.message_count": len(conversation_messages)
                }
            ) as session_span:
                
                # Determine workflow type using router agent
                workflow_type = await self._determine_workflow_type(
                    user_id, 
                    latest_message, 
                    conversation_messages
                )
                
                self.logger.info(f"Router determined workflow type: {workflow_type}")
                session_span.set_attribute("workflow.type", workflow_type)
                
                if workflow_type == "NEW_REQUEST":
                    # Sequential workflow for new mortgage requests
                    response = await self._handle_new_request_conversation(
                        user_id, 
                        latest_message, 
                        conversation_messages,
                        session_id
                    )
                else:
                    # Handoff workflow for existing mortgage requests
                    response = await self._handle_existing_request_conversation(
                        user_id,
                        latest_message,
                        conversation_messages,
                        session_id
                    )
                
                session_span.set_attribute("response.agent", response.get('name', 'unknown'))
                session_span.set_attribute("response.length", len(response.get('content', '')))
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error processing mortgage conversation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': f'I encountered an error while processing your mortgage request. Please try again.'
            }
        finally:
            clear_conversation_context()
    
    async def _determine_workflow_type(
        self, 
        user_id: str, 
        latest_message: str,
        conversation_messages: list
    ) -> str:
        """
        Use the router agent to determine if this is a new or existing request.
        
        Args:
            user_id: The user identifier
            latest_message: The latest user message
            conversation_messages: Full conversation history
            
        Returns:
            "NEW_REQUEST" or "EXISTING_REQUEST"
        """
        # First, check if this is a continuation of a new request flow
        # (user responding to profile selection prompt)
        message_lower = latest_message.lower().strip()
        if message_lower in ['pete', 'bradley']:
            # Check if previous assistant message was the profile selection prompt
            for msg in reversed(conversation_messages[:-1]):  # Skip the current user message
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '').lower()
                    if 'select which profile' in content or 'pete' in content and 'bradley' in content:
                        self.logger.info("Detected continuation of new request flow (profile selection)")
                        return "NEW_REQUEST"
                    break  # Only check the last assistant message
        
        # Check if conversation context mentions existing request details
        context_mentions_request = False
        for msg in conversation_messages:
            content = msg.get('content', '').lower()
            if any(kw in content for kw in ['request_id', 'mr-', 'status:', 'classification:', 'documents:']):
                context_mentions_request = True
                break
        
        # Also check if user has existing requests
        try:
            existing_requests = self.get_request_list(user_id)
            has_existing_requests = len(existing_requests) > 0
        except Exception:
            has_existing_requests = False
        
        # Build context for router
        router_context = f"User message: {latest_message}"
        if context_mentions_request:
            router_context += "\n[Context contains mortgage request details]"
        if has_existing_requests:
            router_context += f"\n[User has {len(existing_requests)} existing mortgage request(s)]"
        
        # Ask router agent to classify
        router_message = ChatMessage(role="user", text=router_context)
        
        try:
            # Use the simpler run method with a new thread
            thread = self._router_agent.get_new_thread()
            response = await self._router_agent.run(router_context, thread=thread)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            self.logger.info(f"Router agent raw response: '{response_text}'")
            response_text = response_text.strip().upper()
            
            if "NEW" in response_text:
                return "NEW_REQUEST"
            elif "EXISTING" in response_text:
                return "EXISTING_REQUEST"
            else:
                # Default based on context
                if context_mentions_request or has_existing_requests:
                    return "EXISTING_REQUEST"
                return "NEW_REQUEST"
                
        except Exception as e:
            self.logger.warning(f"Router agent error, defaulting to context-based decision: {e}")
            # Fallback: determine based on keywords
            message_lower = latest_message.lower()
            new_keywords = ['new mortgage', 'apply', 'start application', 'submit', 'new application']
            if any(kw in message_lower for kw in new_keywords):
                return "NEW_REQUEST"
            return "EXISTING_REQUEST" if (context_mentions_request or has_existing_requests) else "NEW_REQUEST"
    
    async def _handle_new_request_conversation(
        self,
        user_id: str,
        latest_message: str,
        conversation_messages: list,
        session_id: str
    ) -> dict:
        """
        Handle conversation for new mortgage requests using sequential workflow.
        
        For new requests, we guide the user through the application process.
        Uses mock data from Pete or Bardely profiles for demonstration.
        
        The sequential workflow runs: classifier → document_extractor → policy_checker
        The policy_checker agent is responsible for producing the final structured JSON
        that gets persisted to CosmosDB.
        """
        self.logger.info("Handling new mortgage request conversation")
        
        # Check if user has selected a mock profile
        selected_profile = self._detect_mock_selection(conversation_messages)
        
        # If no profile selected yet, prompt user to choose
        if not selected_profile:
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': (
                    "I'd be happy to help you with a new mortgage application! 🏠\n\n"
                    "**For demontration, we'll use mocked application data.**\n\n"
                    "Please select which profile to use:\n\n"
                    "1. **Pete** - Swiss citizen, CHF 850,000 property, 25% down payment\n"
                    "2. **Bradley** - Non-EU citizen, CHF 1,200,000 property, 30% down payment\n\n"
                    "Just type **Pete** or **Bradley** to proceed with the selected profile's documents and data."
                )
            }
        
        self.logger.info(f"Using mock profile: {selected_profile}")
        
        # Load documents and form data for the selected profile
        documents, form_data = self._load_mock_documents(selected_profile)
        
        if not documents:
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': (
                    f"⚠️ Could not find documents for profile '{selected_profile}'.\n\n"
                    f"Please ensure the mock data folder exists at: `data/mortgage/{selected_profile}/`\n"
                    "and contains the required document files (PDF, JPG, PNG, etc.)."
                )
            }
        
        # Create a new mortgage request with the mock data
        request_id = f"MR-{dt.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        
        try:
            # Create and persist the initial request
            self.create_request(
                user_id=user_id,
                form_data=form_data,
                documents={name: url for name, url in documents.items()},
                request_id=request_id
            )
            
            # Prepare workflow input - instruct the policy_checker to output the final structured JSON
            workflow_input = f"""Process this mortgage application through all steps and produce a final structured result.

**Request ID:** {request_id}
**Applicant Profile:** {selected_profile.title()}

**Application Data:**
{json.dumps(form_data, indent=2)}

**Submitted Documents ({len(documents)} files):**
{json.dumps(list(documents.keys()), indent=2)}

**Processing Steps:**
1. CLASSIFIER: Classify each document by category
2. DOCUMENT EXTRACTOR: Extract key data from documents
3. POLICY CHECKER: Check compliance and produce final result

**IMPORTANT - FINAL OUTPUT FORMAT:**
The policy_checker agent MUST produce a JSON object with this exact structure for CosmosDB persistence:

```json
{{
  "classification": {{"filename": "category", ...}},
  "extracted": {{
    "rule_checks": [
      {{"rule": "rule name", "status": "pass|fail|warning", "details": "explanation"}},
      ...
    ]
  }},
  "decision": {{
    "approved": true|false,
    "reasons": ["reason1", "reason2", ...]
  }},
  "status": "COMPLETED"
}}
```

Process the application now."""

            # Run the sequential workflow - just capture the final response
            final_response = ""
            self.logger.info("Starting sequential workflow...")
            
            async for event in self._sequential_workflow.run_stream(workflow_input):
                # We only care about the final output from the last agent
                if isinstance(event, WorkflowOutputEvent):
                    if event.data:
                        for msg in reversed(event.data):
                            if hasattr(msg, 'role') and msg.role == Role.ASSISTANT:
                                final_response = msg.text or ""
                                self.logger.info(f"Captured final workflow output (length: {len(final_response)})")
                                break
            
            self.logger.info(f"Workflow completed. Final response length: {len(final_response)}")
            
            # Parse the final structured JSON from policy_checker's response
            result_json = self._parse_agent_json_response(final_response)
            
            if result_json:
                # Persist the structured result to CosmosDB
                updates = {}
                
                if 'classification' in result_json:
                    updates['classification'] = result_json['classification']
                
                if 'extracted' in result_json:
                    updates['extracted'] = result_json['extracted']
                elif 'rule_checks' in result_json:
                    updates['extracted'] = {'rule_checks': result_json['rule_checks']}
                
                if 'decision' in result_json:
                    updates['decision'] = result_json['decision']
                elif 'approved' in result_json:
                    updates['decision'] = {
                        'approved': result_json['approved'],
                        'reasons': result_json.get('reasons', [])
                    }
                
                updates['status'] = result_json.get('status', 'COMPLETED')
                
                # Persist to CosmosDB
                if updates:
                    self.update_request(user_id, request_id, updates)
                    self.logger.info(f"Persisted workflow results for {request_id}: {list(updates.keys())}")
                
                # Build human-readable response
                return self._build_workflow_response(request_id, selected_profile, documents, updates)
            else:
                # Could not parse structured output - return the raw response
                self.logger.warning("Could not parse structured JSON from workflow output")
                self.update_request(user_id, request_id, {'status': 'PENDING_REVIEW'})
                
                return {
                    'role': 'assistant',
                    'name': 'policy-checker',
                    'content': final_response or f"Your mortgage request ({request_id}) has been created but requires manual review."
                }
            
        except Exception as e:
            self.logger.error(f"Error in sequential workflow: {e}")
            import traceback
            traceback.print_exc()
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': f'I encountered an issue processing your new mortgage application: {str(e)}'
            }
    
    def _build_workflow_response(
        self,
        request_id: str,
        profile: str,
        documents: dict,
        updates: dict
    ) -> dict:
        """Build a human-readable response from the workflow results."""
        
        # Build classification summary
        classification_info = ""
        if updates.get('classification'):
            classification_info = "\n\n**Document Classification:**\n"
            for filename, category in updates['classification'].items():
                classification_info += f"- {filename}: {category}\n"
        
        # Build decision info with rule checks
        decision_info = ""
        decision = updates.get('decision', {})
        if decision:
            approved = decision.get('approved', False)
            reasons = decision.get('reasons', [])
            decision_info = f"\n\n**Decision:** {'✅ APPROVED' if approved else '❌ NOT APPROVED'}\n"
            if reasons:
                decision_info += f"**Reasons:** {'; '.join(reasons)}\n"
            
            # Add formatted rule checks
            rule_checks = updates.get('extracted', {}).get('rule_checks', [])
            if rule_checks:
                decision_info += f"\n{self._format_rule_checks_for_display(rule_checks)}"
        
        summary = (
            f"✅ **Mortgage Request Processed**\n\n"
            f"**Request ID:** `{request_id}`\n"
            f"**Profile:** {profile.title()}\n"
            f"**Documents Processed:** {len(documents)}\n"
            f"**Status:** {updates.get('status', 'COMPLETED')}"
            f"{classification_info}"
            f"{decision_info}"
        )
        
        return {
            'role': 'assistant',
            'name': 'policy-checker',
            'content': summary
        }

    async def _handle_existing_request_conversation(
        self,
        user_id: str,
        latest_message: str,
        conversation_messages: list,
        session_id: str
    ) -> dict:
        """
        Handle conversation for existing mortgage requests using handoff workflow.
        
        Uses the handoff workflow with coordinator and specialist agents to help
        users with their ongoing mortgage applications.
        
        Args:
            user_id: The user identifier
            latest_message: The latest user message
            conversation_messages: Full conversation history
            session_id: Session/chat ID for tracing
            
        Returns:
            Dict with 'role', 'name', 'content' keys representing the agent's reply
        """
        self.logger.info("Handling existing mortgage request conversation")
        
        # Set up context for query functions so they can access user data
        set_query_context(self._persister, user_id)
        
        try:
            # Convert conversation history to ChatMessage objects for the workflow
            chat_messages = []
            for msg in conversation_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                author_name = msg.get('name')
                
                if content:
                    chat_msg = ChatMessage(
                        role=role,
                        text=content,
                        author_name=author_name
                    )
                    chat_messages.append(chat_msg)
            
            if not chat_messages:
                return {
                    'role': 'assistant',
                    'name': 'mortgage-coordinator',
                    'content': "I didn't receive a message. How can I help you with your mortgage application?"
                }
            
            self.logger.info(f"Running handoff workflow with {len(chat_messages)} messages")
            
            final_response = ""
            responding_agent = "mortgage-coordinator"
            
            async for event in self._handoff_workflow.run_stream(chat_messages):
                if isinstance(event, RequestInfoEvent):
                    # Handle HandoffAgentUserRequest with agent_response
                    if hasattr(event.data, 'agent_response') and event.data.agent_response:
                        agent_response = event.data.agent_response
                        if hasattr(agent_response, 'text') and agent_response.text:
                            final_response = agent_response.text
                            responding_agent = getattr(event, 'source_executor_id', 'mortgage-coordinator')
                
                elif isinstance(event, WorkflowOutputEvent):
                    if event.data:
                        for msg in reversed(event.data):
                            if hasattr(msg, 'role') and msg.role == Role.ASSISTANT:
                                final_response = msg.text or final_response
                                responding_agent = getattr(msg, 'author_name', responding_agent)
                                break
                
                elif isinstance(event, ExecutorCompletedEvent):
                    if event.data is not None:
                        if hasattr(event.data, 'text') and event.data.text:
                            final_response = event.data.text
                            responding_agent = event.executor_id or responding_agent
            
            self.logger.info(f"Handoff workflow response: {len(final_response)} chars from '{responding_agent}'")
            
            return {
                'role': 'assistant',
                'name': responding_agent,
                'content': final_response or 'I apologize, but I was unable to generate a response about your mortgage request.'
            }
            
        except Exception as e:
            self.logger.error(f"Error in handoff workflow: {e}")
            import traceback
            traceback.print_exc()
            return {
                'role': 'assistant',
                'name': 'mortgage-coordinator',
                'content': f'I encountered an issue while handling your mortgage request: {str(e)}'
            }
        finally:
            # Clean up query context
            clear_query_func_context()
    
    def _get_mock_data_folder(self) -> Path:
        """Get the path to the mock mortgage data folder."""
        # Navigate from orchestrators -> foundry -> backend -> src -> data/mortgage
        return Path(__file__).parent.parent.parent.parent / "data" / "mortgage"
    
    def _detect_mock_selection(self, conversation_messages: list) -> str | None:
        """
        Detect if user has selected Pete or Bardely mock data from conversation.
        
        Returns:
            'pete', 'bardely', or None if no selection detected
        """
        for msg in reversed(conversation_messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                if 'pete' in content:
                    return 'pete'
                if 'bradley' in content:
                    return 'bardely'
        return None
    
    def _load_mock_documents(self, profile_name: str) -> tuple[dict[str, str], dict]:
        """
        Load mock documents from the data/mortgage/{profile_name} folder.
        
        Args:
            profile_name: 'pete' or 'bradley'
            
        Returns:
            Tuple of (documents dict with base64 content, mock form data)
        """
        mock_folder = self._get_mock_data_folder() / profile_name
        
        if not mock_folder.exists():
            self.logger.warning(f"Mock data folder not found: {mock_folder}")
            return {}, {}
        
        documents = {}
        for file_path in mock_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.doc', '.docx']:
                try:
                    with open(file_path, 'rb') as f:
                        content = base64.b64encode(f.read()).decode('utf-8')
                    documents[file_path.name] = f"data:application/octet-stream;base64,{content}"
                    self.logger.info(f"Loaded mock document: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path.name}: {e}")
        
        # Define mock form data for each profile
        mock_form_data = {
            'pete': {
                'property_value': 850000,
                'income': 180000,
                'liabilities': 15000,
                'down_payment_pc': 25,
                'swiss_or_eu_citizen': True,
                'pledge_p2': False,
                'pledge_p2_amount': 0,
                'pledge_p3': True,
                'pledge_p3_amount': 50000,
            },
            'bardely': {
                'property_value': 1200000,
                'income': 250000,
                'liabilities': 30000,
                'down_payment_pc': 30,
                'swiss_or_eu_citizen': False,
                'pledge_p2': True,
                'pledge_p2_amount': 80000,
                'pledge_p3': True,
                'pledge_p3_amount': 40000,
            }
        }
        
        return documents, mock_form_data.get(profile_name, {})
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._credential:
            await self._credential.close()
            self._credential = None
        self._chat_client = None
        self._sequential_workflow = None
        self._handoff_workflow = None
        self._router_agent = None
        self._initialized = False
        self.logger.info("Mortgage Orchestrator closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def interactive_human_feedback(agent_response: str) -> str | None:
    """
    Interactive human feedback callback for CLI usage.
    
    Args:
        agent_response: The agent's response to review
        
    Returns:
        Feedback string if changes needed, None to approve
    """
    print("\n" + "=" * 60)
    print("🔍 HUMAN REVIEW REQUESTED")
    print("=" * 60)
    print(f"\nAgent Response:\n{agent_response[:1000]}...")
    print("\n" + "-" * 60)
    
    user_input = input("Enter feedback (or press Enter to approve): ").strip()
    
    if user_input:
        return user_input
    return None


async def main():
    """Main function to demonstrate the Mortgage Orchestrator."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🏠 Moneta Bank Mortgage Processing System")
    print("=" * 50)
    
    # Create a sample mortgage request
    sample_request = MortgageRequest(
        request_id="MR-2024-001",
        data={
            "property_value": 1000000,
            "income": 200000,
            "liabilities": 20000,
            "down_payment_pc": 25,
            "pledge_p2": False,
            "pledge_p2_amount": 0,
            "pledge_p3": False,
            "pledge_p3_amount": 0
        },
        documents={
            "id_document": "https://example.com/documents/id_passport.pdf",
            "income_proof": "https://example.com/documents/payslip.pdf",
            "bank_statements": "https://example.com/documents/bank_statement.pdf",
            "property_valuation": "https://example.com/documents/valuation.pdf"
        }
    )
    
    print(f"\n📋 Processing Mortgage Request: {sample_request.request_id}")
    print(f"   Property Value: £{sample_request.data['property_value']:,}")
    print(f"   Documents: {len(sample_request.documents)}")
    
    try:
        # Process with human review enabled
        async with MortgageOrchestrator(enable_human_review=True) as orchestrator:
            result = await orchestrator.process_mortgage_request(
                sample_request,
                human_feedback_callback=interactive_human_feedback
            )
            
            print("\n" + "=" * 60)
            print("📊 PROCESSING COMPLETE")
            print("=" * 60)
            print(f"\nRequest ID: {result['request_id']}")
            print(f"Status: {result['status']}")
            print(f"Recommendation: {result['decision']['recommendation']}")
            print(f"\nReasoning:\n{result['decision']['reasoning'][:500]}...")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
