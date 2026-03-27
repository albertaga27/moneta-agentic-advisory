"""
Policy Check Agent - Microsoft Agent Framework Implementation
Verifies mortgage applications against regulatory requirements and lending policies.
"""

import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Microsoft Agent Framework imports
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential

# Import policy check functions
from .policy_check_functions import policy_check_functions


POLICY_CHECK_AGENT_INSTRUCTIONS = """You are a Regulatory Compliance Officer for Moneta Bank's Mortgage Division in Switzerland.

**Your Role:**
As the FINAL agent in the mortgage processing workflow, you must:
1. Verify mortgage applications against Swiss regulatory requirements
2. Compile and output the FINAL structured JSON result for CosmosDB persistence

**Swiss Mortgage Regulatory Framework You Apply:**

1. **Down-payment Requirement** - Minimum 20% of property value as down payment

2. **Hard-Equity Requirement** - Minimum 10% must be "hard equity" (own funds)
   - Pillar 2 and Pillar 3 pension pledges do NOT count as hard equity

3. **Loan-to-Value (LTV) Limit** - Maximum 80% LTV for initial mortgage

4. **Affordability** - Total housing costs ≤ 33% of gross income
   - Calculated at 5% stress test interest rate
   - Includes: interest + 1% maintenance + amortisation

5. **Amortisation** - Second mortgage must be repaid within 15 years
   - Target LTV of 66.7% within 15 years

6. **Lex Koller Compliance** - Restrictions on property purchases by non-Swiss/non-EU citizens

7. **Cross-border Compliance** - Residence permit requirements for non-Swiss applicants

8. **Document Completeness** - Required: identity, proof_of_income, credit_history, property_valuation

**Instructions:**
1. Review the classification results from the previous agents
2. Use the 'check_policy_compliance' function to perform the compliance check
3. Compile ALL results into the final JSON structure

**CRITICAL - You MUST output the final result as a JSON object with this EXACT structure:**

```json
{
  "classification": {
    "document_filename.pdf": "category",
    ...
  },
  "extracted": {
    "rule_checks": [
      {"rule": "Rule Name", "status": "pass|fail|warning", "details": "..."},
      ...
    ]
  },
  "decision": {
    "approved": true|false,
    "reasons": ["reason1", "reason2", ...]
  },
  "status": "COMPLETED"
}
```

**Where:**
- `classification`: Copy the document classification from the classifier agent's results
- `extracted.rule_checks`: Array of all rule check results from check_policy_compliance
- `decision.approved`: true if ALL mandatory checks passed, false otherwise
- `decision.reasons`: Summary list of key reasons for the decision
- `status`: Always "COMPLETED" once processing is done

Do NOT output any other text - ONLY output the final JSON object."""


async def create_policy_check_agent(chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """
    Create a Policy Check agent using Microsoft Agent Framework.
    
    Args:
        chat_client: The Azure OpenAI chat client
        
    Returns:
        ChatAgent configured for policy compliance checking
    """
    agent = chat_client.as_agent(
        instructions=POLICY_CHECK_AGENT_INSTRUCTIONS,
        name="policy_check_agent",
        tools=policy_check_functions
    )
    
    print(f"✅ Created Policy Check agent: {agent.name}")
    return agent


# Agent configuration for registration
POLICY_CHECK_AGENT_CONFIG = {
    "name": "mortgage-policy-check-agent",
    "instructions": POLICY_CHECK_AGENT_INSTRUCTIONS,
    "description": "Policy Check Agent - verifies mortgage applications against regulatory requirements",
    "tools": policy_check_functions
}


async def main(): 
    """
    Main function to initialize and run the Policy Check agent standalone.
    """
    os.system('cls' if os.name=='nt' else 'clear')
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if not endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT environment variable is not set")
        return

    try:
        async with AzureCliCredential() as credential:
            chat_client = AzureOpenAIChatClient(
                endpoint=endpoint,
                deployment_name=deployment_name,
                credential=credential
            )
            
            agent = await create_policy_check_agent(chat_client)
            
            print(f"\n⚖️ You're chatting with: {agent.name}")
            print("This agent verifies mortgage applications against regulatory requirements.")
            print("Type 'quit' to exit.\n")

            # Sample application for testing
            sample_app = {
                "data": {
                    "applicant_name": "John Smith",
                    "applicant_income": 75000,
                    "applicant_dob": "1985-03-15",
                    "employment_status": "employed",
                    "property_address": "123 Main St, London",
                    "property_value": 450000,
                    "loan_amount": 360000,
                    "loan_term_years": 25,
                    "loan_purpose": "purchase",
                    "credit_score": 720,
                    "interest_rate": 5.5
                },
                "documents": {
                    "id_document": "path/to/id.pdf",
                    "income_proof": "path/to/payslip.pdf",
                    "bank_statements": "path/to/statements.pdf"
                },
                "extracted_data": {
                    "id_document": {
                        "date_of_birth": {"value": "1985-03-15"}
                    }
                }
            }
            
            print("Sample application loaded. Ask me to check compliance!")

            while True:
                user_prompt = input("\nEnter your request (or type 'quit' to exit): ")
                
                if user_prompt.lower() in ["quit", "exit", "q"]:
                    print("👋 Thank you for using the Policy Check service!")
                    break
                    
                if not user_prompt.strip():
                    print("Please enter a valid request.")
                    continue

                try:
                    import json
                    # If user asks to check, include the sample data
                    if "check" in user_prompt.lower() or "verify" in user_prompt.lower() or "compliance" in user_prompt.lower():
                        user_prompt += f"\n\nApplication data:\n{json.dumps(sample_app, indent=2)}"
                    
                    thread = agent.get_new_thread()
                    response = await agent.run(user_prompt, thread=thread)
                    print(f"\n⚖️ Policy Check Agent Response:\n{response.text}")
                        
                except Exception as e:
                    print(f"❌ Error processing request: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to initialize Policy Check agent: {str(e)}")


if __name__ == '__main__': 
    asyncio.run(main())
