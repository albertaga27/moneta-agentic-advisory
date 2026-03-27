# Insurance Configuration
INS_PREDEFINED_QUESTIONS = [
    "Provide information about my client John Doe",
    "Can he travel to Bali with his current coverage?",
    "Search for detailed terms and conditions for coverages abroad",
    "Is COVID-19 treatments in Indonesia covered by travel insurance international?"
]

INS_AGENTS = {
    'ins-crm-agent': {
        'emoji': '👥', 
        'color': '#17a2b8',
        'description': 'Access and retrieve customer information from the company\'s database. It specifically focuses on insurance-related client data, such as policies, coverages, and premiums, but will only provide information when given a specific client\'s name or ID.'
    },
    'ins-policies-agent': {
        'emoji': '🔍', 
        'color': '#ffc107',
        'description': 'Dedicated tool for answering general questions about investment funds and ETFs. Unlike the CRM Agent which handles client-specific data, this agent responds to general product inquiries by searching through an AI-powered knowledge base. It\'s designed to provide accurate product information, details about fund offerings, and general investment conditions.'
    }
}

# Banking Configuration
BANK_PREDEFINED_QUESTIONS = [
    "Provide me a summary of the portfolio's positions of my client Pete Mitchell",
    "What are our Chief Investment Office (CIO) believes on the AI sector?",
    "What is our in-house view from our CIO about Growth investing?",
    "Show Pete Mitchell portfolio performance and suggest any rebalancing options based on recent CIO views.",
    "List 3 Funds or ETFs we are offering for growth focused strategies",
    "Can you give me an update on the UBS 100 Index Switzerland Equity Fund CHF and its latest performance?",
    "Craft a rebalance proposal for the client Pete Mitchell increasing the weight of investments in tech stocks absed on our offering"

]

BANK_AGENTS = {
    'bank-crm-agent': {
        'emoji': '👥', 
        'color': '#17a2b8',
        'description': 'Access and retrieve customer information from the company\'s database. It specifically focuses on insurance-related client data, such as policies, coverages, and premiums, but will only provide information when given a specific client\'s name or ID.'
    },
    'bank-funds-agent': {
        'emoji': '💰', 
        'color': '#007bff',
        'description': 'Provides fund analysis and investment recommendations. It searches through internal bank documents to retrieve the institution\'s official investment views, research findings, and recommendations. This agent is designed to help advisors make informed decisions about their clients\' investments.'
    },
    'bank-cio-agent': {
        'emoji': '📈', 
        'color': '#ffc107',
        'description': 'Provides access to official investment research and analysis from Chief Investment Office (CIO). It searches through internal bank documents to retrieve the institution\'s official investment views, research findings, and recommendations.'
    },
    'bank-news-agent': {
        'emoji': '📰', 
        'color': '#6c757d',
        'description': 'Automatically collects and organizes the latest investment news from Morningstar for your portfolio positions. It helps you stay informed about your investments by gathering relevant articles, including their titles, descriptions, categories, and publication details, all presented in an organized format for easy review.'
    }
}

# Mortgage Configuration
MORTGAGE_PREDEFINED_QUESTIONS = [
    "Show me my mortgage applications",
    "What's the status of the latest mortgage request?",
    "Check if my latest mortgage request application is approved",
    "What documents are needed to submit a new mortgage request?",
    "Start the process for a new mortgage request",
    "What are the policy requirements for mortgage approval?"
]

MORTGAGE_AGENTS = {
    'mortgage-coordinator': {
        'emoji': '🏠',
        'color': '#28a745',
        'description': 'Main coordinator for mortgage applications. Routes requests to specialists and helps you find your mortgage requests by ID, status, or other criteria. Use this agent to get an overview of your applications.'
    },
    'classifier': {
        'emoji': '📋',
        'color': '#17a2b8',
        'description': 'Document Classification Specialist. Analyzes and categorizes your mortgage documents (ID, income proof, property valuation, etc.) to ensure all required documentation is properly organized.'
    },
    'document_extractor': {
        'emoji': '📄',
        'color': '#ffc107',
        'description': 'Document Extraction Specialist. Extracts key data from your submitted documents using AI-powered document intelligence, including income figures, property details, and identity information.'
    },
    'policy_checker': {
        'emoji': '✅',
        'color': '#dc3545',
        'description': 'Policy Compliance Specialist. Verifies your mortgage application against regulatory requirements and bank policies, providing approval recommendations and explaining any issues.'
    }
}


# CSS Styles
AGENT_STYLES = """
    <style>
    .agent-list {
        margin: 10px 0;
        padding: 0;
    }
    .agent-item {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
        position: relative;
    }
    .agent-item:hover .agent-tooltip {
        display: block;
    }
    .agent-emoji {
        font-size: 20px;
        margin-right: 10px;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }
    .agent-name {
        flex-grow: 1;
        font-size: 16px;
    }
    .agent-status {
        color: #4CAF50;
        font-size: 14px;
    }
    .agent-tooltip {
        display: none;
        position: absolute;
        background-color: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        width: 200px;
        left: 50%;
        transform: translateX(-50%);
        top: 100%;
        z-index: 1000;
        margin-top: 5px;
    }
    .agent-tooltip::before {
        content: '';
        position: absolute;
        top: -5px;
        left: 50%;
        transform: translateX(-50%);
        border-width: 0 5px 5px 5px;
        border-style: solid;
        border-color: transparent transparent rgba(0, 0, 0, 0.9) transparent;
    }
    
    /* Conversation button styles */
    .stButton>button {
        height: auto !important;
        text-align: left !important;
    }
    .conversation-title {
        font-weight: 500;
    }
    .conversation-meta {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.6);
    }
    </style>
"""

# General styles
GENERAL_STYLES = """
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        width: auto;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-image: url('resources/insurance_logo.png');
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        background-attachment: fixed;
    }
    .login-box {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
"""