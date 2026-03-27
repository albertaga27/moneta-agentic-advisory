"""
Document Extraction Agent - Microsoft Agent Framework Implementation
Extracts data from mortgage documents using Azure Document Intelligence.
"""

import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Microsoft Agent Framework imports
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential

# Import document extraction functions
from .document_extraction_functions import document_extraction_functions


DOCUMENT_EXTRACTION_AGENT_INSTRUCTIONS = """You are a Document Extraction Specialist for Moneta Bank's Mortgage Processing System.

**Your Role:**
Extract and validate information from mortgage-related documents using Azure Document Intelligence.

**Document Types You Handle:**
1. **ID Documents**: Passports, driver's licenses, ID cards
   - Extract: Full name, date of birth, document number, expiry date, nationality

2. **Income Proof**: Payslips, employment letters, P60s, tax returns
   - Extract: Employer name, salary, pay period, employment start date, position

3. **Bank Statements**: Monthly account statements
   - Extract: Account holder, account number, statement period, balances, transaction patterns

4. **Property Valuations**: Property survey and valuation reports
   - Extract: Property address, estimated value, property type, condition, valuation date

**Instructions:**
- Use the 'extract_document_data' function to process each document
- Verify extracted data has reasonable confidence scores (>0.80 preferred)
- Flag any fields with low confidence for manual review
- Cross-reference extracted data with application data when available
- Report any discrepancies between documents and application data
- Be CONCISE and focus on the extracted data
- If extraction fails, clearly explain why and what manual steps are needed

**Quality Checks:**
- Ensure document is not expired (for ID documents)
- Verify dates are in valid format
- Check for consistency across documents (e.g., name matches)"""


async def create_document_extraction_agent(chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """
    Create a Document Extraction agent using Microsoft Agent Framework.
    
    Args:
        chat_client: The Azure OpenAI chat client
        
    Returns:
        ChatAgent configured for document extraction
    """
    agent = chat_client.as_agent(
        instructions=DOCUMENT_EXTRACTION_AGENT_INSTRUCTIONS,
        name="document_extraction_agent",
        tools=document_extraction_functions
    )
    
    print(f"✅ Created Document Extraction agent: {agent.name}")
    return agent


# Agent configuration for registration
DOCUMENT_EXTRACTION_AGENT_CONFIG = {
    "name": "mortgage-document-extraction-agent",
    "instructions": DOCUMENT_EXTRACTION_AGENT_INSTRUCTIONS,
    "description": "Document Extraction Agent - extracts data from mortgage documents using Azure Document Intelligence",
    "tools": document_extraction_functions
}


async def main(): 
    """
    Main function to initialize and run the Document Extraction agent standalone.
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
            
            agent = await create_document_extraction_agent(chat_client)
            
            print(f"\n📄 You're chatting with: {agent.name}")
            print("This agent extracts data from mortgage documents.")
            print("Type 'quit' to exit.\n")
            
            print("Example requests:")
            print("- 'Extract data from the ID document at /path/to/id.pdf'")
            print("- 'Process the income proof document'")
            print("- 'Extract property valuation details from https://example.com/valuation.pdf'")

            while True:
                user_prompt = input("\nEnter your request (or type 'quit' to exit): ")
                
                if user_prompt.lower() in ["quit", "exit", "q"]:
                    print("👋 Thank you for using the Document Extraction service!")
                    break
                    
                if not user_prompt.strip():
                    print("Please enter a valid request.")
                    continue

                try:
                    thread = agent.get_new_thread()
                    response = await agent.run(user_prompt, thread=thread)
                    print(f"\n📄 Document Extraction Agent Response:\n{response.text}")
                        
                except Exception as e:
                    print(f"❌ Error processing request: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to initialize Document Extraction agent: {str(e)}")


if __name__ == '__main__': 
    asyncio.run(main())
