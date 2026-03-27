# Mortgage Processing Agents

This module provides a complete mortgage application processing system using Microsoft Agent Framework with Sequential orchestration and Human-in-the-Loop (HIL) capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MORTGAGE ORCHESTRATOR                                 │
│                    (Sequential Workflow + Human-in-Loop)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐     ┌─────────────────────┐   ┌─────────────────┐
    │  CLASSIFIER   │     │ DOCUMENT EXTRACTOR  │   │  POLICY CHECK   │
    │    AGENT      │────▶│       AGENT         │──▶│     AGENT       │
    │               │     │                     │   │                 │
    │ - App Type    │     │ - Azure Doc Intel   │   │ - LTV Check     │
    │ - Completeness│     │ - ID Documents      │   │ - DTI Check     │
    │ - Risk Level  │     │ - Income Proof      │   │ - Credit Score  │
    │               │     │ - Bank Statements   │   │ - Affordability │
    └───────────────┘     │ - Property Valuation│   │ - Age Limits    │
                          └─────────────────────┘   └─────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  HUMAN REVIEW       │
                          │  (Optional HIL)     │
                          │                     │
                          │ - Verify extracted  │
                          │   data accuracy     │
                          │ - Provide feedback  │
                          └─────────────────────┘
```

## Components

### 1. MortgageRequest Model (`models.py`)

Data class representing a mortgage application:

```python
from mortgage.models import MortgageRequest

request = MortgageRequest(
    request_id="MR-2024-001",
    data={
        "applicant_name": "John Smith",
        "applicant_income": 75000,
        "property_value": 450000,
        "loan_amount": 360000,
        "loan_term_years": 25,
        "credit_score": 720,
        # ... other fields
    },
    documents={
        "id_document": "path/to/id.pdf",
        "income_proof": "path/to/payslip.pdf",
        # ... other documents
    }
)
```

### 2. Classifier Agent (`classifier_agent.py`)

Classifies mortgage applications by:
- **Application Type**: purchase, refinance, equity_release, construction
- **Completeness**: complete, partial, incomplete
- **Risk Category**: low, medium, high

### 3. Document Extraction Agent (`document_extraction_agent.py`)

Extracts data from mortgage documents using Azure Document Intelligence:
- ID Documents (passport, driver's license)
- Income Proof (payslips, employment letters)
- Bank Statements
- Property Valuations

### 4. Policy Check Agent (`policy_check_agent.py`)

Verifies applications against regulatory requirements:
- Loan-to-Value (LTV) ratio limits
- Debt-to-Income (DTI) ratio limits
- Credit score requirements
- Age eligibility
- Affordability stress testing
- Documentation completeness

### 5. Mortgage Orchestrator (`mortgage_orchestrator.py`)

Sequential workflow orchestrator with Human-in-the-Loop:

```python
from mortgage import MortgageOrchestrator, MortgageRequest

async def process_application():
    orchestrator = MortgageOrchestrator(
        enable_human_review=True,
        human_review_agents=["document_extractor"]  # Pause after document extraction
    )
    
    await orchestrator.initialize()
    
    result = await orchestrator.process_mortgage_request(
        mortgage_request,
        human_feedback_callback=my_callback
    )
    
    await orchestrator.close()
```

## Human-in-the-Loop (HIL)

The orchestrator supports human review at configurable points in the workflow:

```python
# Define a human feedback callback
async def human_feedback_callback(agent_response: str) -> str | None:
    """
    Called when workflow pauses for human review.
    
    Args:
        agent_response: The agent's output to review
        
    Returns:
        None to approve, or feedback string to request iteration
    """
    # Display response to human reviewer
    print(f"Agent Response: {agent_response}")
    
    # Get human input
    feedback = input("Enter feedback or press Enter to approve: ")
    
    return feedback if feedback else None

# Use in orchestrator
result = await orchestrator.process_mortgage_request(
    request,
    human_feedback_callback=human_feedback_callback
)
```

## Configuration

### Environment Variables

```bash
# Azure OpenAI (for agent LLM)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Azure Document Intelligence (for document extraction)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intel.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key  # Optional, can use DefaultAzureCredential
```

### Regulatory Limits

Default regulatory limits are defined in `policy_check_functions.py`:

```python
REGULATORY_LIMITS = {
    "max_ltv_ratio": 95.0,          # Maximum LTV %
    "max_dti_ratio": 4.5,           # Maximum DTI multiplier
    "min_credit_score": 580,        # Minimum credit score
    "max_loan_term_years": 40,      # Maximum loan term
    "min_applicant_age": 18,        # Minimum age
    "max_applicant_age_at_term_end": 75,  # Max age at mortgage end
    "stress_test_rate_buffer": 3.0, # Interest rate stress buffer %
}
```

## Usage Examples

### Standalone Agent Testing

Each agent can be run independently for testing:

```bash
# Test classifier agent
python -m foundry.agents.banking.mortgage.classifier_agent

# Test document extraction agent
python -m foundry.agents.banking.mortgage.document_extraction_agent

# Test policy check agent
python -m foundry.agents.banking.mortgage.policy_check_agent
```

### Full Workflow

```bash
# Run the complete mortgage orchestrator
python -m foundry.agents.banking.mortgage.mortgage_orchestrator
```

### Programmatic Usage

```python
import asyncio
from foundry.agents.banking.mortgage import (
    MortgageOrchestrator,
    MortgageRequest
)

async def main():
    request = MortgageRequest(
        request_id="MR-2024-001",
        data={
            "applicant_name": "John Smith",
            "applicant_income": 75000,
            "property_value": 450000,
            "loan_amount": 360000,
            "loan_term_years": 25,
            "credit_score": 720,
            "loan_purpose": "purchase",
            "employment_status": "employed"
        },
        documents={
            "id_document": "https://storage.example.com/id.pdf",
            "income_proof": "https://storage.example.com/payslip.pdf",
            "bank_statements": "https://storage.example.com/statements.pdf"
        }
    )
    
    async with MortgageOrchestrator() as orchestrator:
        result = await orchestrator.process_mortgage_request(request)
        
        print(f"Status: {result['status']}")
        print(f"Recommendation: {result['decision']['recommendation']}")
        print(f"Reasoning: {result['decision']['reasoning']}")

asyncio.run(main())
```

## Workflow States

The `MortgageRequest.status` field tracks the application state:

| Status | Description |
|--------|-------------|
| `RECEIVED` | Initial state when request is created |
| `PROCESSING` | Workflow is actively processing |
| `CLASSIFYING` | Classifier agent is analyzing |
| `EXTRACTING` | Document extraction in progress |
| `POLICY_CHECK` | Policy compliance check in progress |
| `PENDING_REVIEW` | Requires human review |
| `APPROVED` | Application approved |
| `REJECTED` | Application rejected |
| `ERROR` | Processing error occurred |

## Decision Recommendations

The policy check agent provides one of three recommendations:

| Recommendation | Criteria |
|---------------|----------|
| `APPROVE` | All mandatory checks pass, no significant warnings |
| `NEEDS_REVIEW` | Mandatory checks pass but warnings exist, or missing info |
| `REJECT` | One or more mandatory compliance checks failed |

## Dependencies

```
agent-framework-azure-ai>=0.1.0
azure-identity>=1.15.0
azure-ai-documentintelligence>=1.0.0  # Optional, for document extraction
python-dotenv>=1.0.0
```

## Installation

```bash
# Install Agent Framework with Azure AI support (preview)
pip install agent-framework-azure-ai --pre

# Install Azure Document Intelligence (optional)
pip install azure-ai-documentintelligence
```

## Testing

The document extraction functions include a mock implementation that activates when Azure Document Intelligence is not configured. This allows testing the workflow without requiring actual document processing capabilities.

## License

Internal use only - Moneta Bank
