"""
Classifier Agent - Microsoft Agent Framework Implementation
Classifies mortgage documents based on filenames to route them for processing.
"""

import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Microsoft Agent Framework imports
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import AzureCliCredential

# Import classifier functions
from .classifier_functions import classifier_functions


CLASSIFIER_AGENT_INSTRUCTIONS = """You are a file-routing assistant for Moneta Bank's Mortgage Processing System.

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

**Instructions:**
- Use the 'classify_documents' function to classify the document list
- You do NOT need to download or read file contents - classify based on filenames only
- For production OCR & page splitting, Azure Document Intelligence should be used
- Reply with valid JSON format: {"classification": {<filename>: <label>, ...}}
- Be CONCISE and provide only the classification results
- If a filename is ambiguous, use your best judgment based on common naming patterns"""


async def create_classifier_agent(chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """
    Create a Classifier agent using Microsoft Agent Framework.
    
    Args:
        chat_client: The Azure OpenAI chat client
        
    Returns:
        ChatAgent configured for mortgage classification
    """
    agent = chat_client.as_agent(
        instructions=CLASSIFIER_AGENT_INSTRUCTIONS,
        name="mortgage_classifier_agent",
        tools=classifier_functions
    )
    
    print(f"✅ Created Classifier agent: {agent.name}")
    return agent


# Agent configuration for registration
CLASSIFIER_AGENT_CONFIG = {
    "name": "mortgage-classifier-agent",
    "instructions": CLASSIFIER_AGENT_INSTRUCTIONS,
    "description": "Mortgage Document Classifier - routes documents by filename to appropriate categories",
    "tools": classifier_functions
}


async def main(): 
    """
    Main function to initialize and run the Classifier agent standalone.
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
            
            agent = await create_classifier_agent(chat_client)
            
            print(f"\n📋 You're chatting with: {agent.name}")
            print("This agent classifies mortgage documents based on filenames.")
            print("Type 'quit' to exit.\n")

            # Sample documents for testing
            sample_documents = {
                "documents": [
                    "https://storage.example.com/mortgage/passport_john_smith.pdf",
                    "https://storage.example.com/mortgage/salary_slip_jan_2024.pdf",
                    "https://storage.example.com/mortgage/betreibungsauszug.pdf",
                    "https://storage.example.com/mortgage/property_valuation_report.pdf",
                    "https://storage.example.com/mortgage/grundbuchauszug_zurich.pdf",
                    "https://storage.example.com/mortgage/kaufvertrag_draft.pdf",
                    "https://storage.example.com/mortgage/building_insurance_policy.pdf",
                    "https://storage.example.com/mortgage/residence_permit_b.pdf"
                ]
            }
            
            print("Sample documents loaded:")
            for doc in sample_documents["documents"]:
                print(f"  - {doc.split('/')[-1]}")
            print("\nAsk me to classify these documents!")

            while True:
                user_prompt = input("\nEnter your request (or type 'quit' to exit): ")
                
                if user_prompt.lower() in ["quit", "exit", "q"]:
                    print("👋 Thank you for using the Classifier service!")
                    break
                    
                if not user_prompt.strip():
                    print("Please enter a valid request.")
                    continue

                try:
                    import json
                    # If user asks to classify, include the sample documents
                    if "classify" in user_prompt.lower():
                        user_prompt += f"\n\nDocuments to classify:\n{json.dumps(sample_documents, indent=2)}"
                    
                    thread = agent.get_new_thread()
                    response = await agent.run(user_prompt, thread=thread)
                    print(f"\n📋 Classifier Agent Response:\n{response.text}")
                        
                except Exception as e:
                    print(f"❌ Error processing request: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to initialize Classifier agent: {str(e)}")


if __name__ == '__main__': 
    asyncio.run(main())
