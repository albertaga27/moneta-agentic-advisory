"""
Classifier functions for mortgage document classification.
This module provides AI-powered classification of mortgage documents based on filenames.

The classifier routes documents to appropriate categories based on filename analysis.
For production use, replace with Azure Document Intelligence for real OCR & page splitting.
"""

import json
import logging
import re
from typing import Annotated, Any, Callable
from pathlib import Path

# Import tracing utilities from backend root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from tracing import get_tracing_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Document classification categories
DOCUMENT_CATEGORIES = [
    "identity",                    # ID, driver license or passport
    "residence_permit",            # Residence permit (if non-Swiss)
    "proof_of_income",             # Salary slips, tax return, account statements
    "credit_history",              # Betreibungsauszug (debt collection extract)
    "property_valuation",          # Listing, appraisal
    "lex_koller_authorisation",    # Lex Koller authorization (for foreign buyers)
    "land_registry_extract",       # Land registry extract
    "building_insurance",          # Building insurance certificate or policy
    "purchase_contract",           # Draft purchase contract or reservation agreement
    "other"                        # Unclassified documents
]

# Keywords for rule-based classification fallback
CLASSIFICATION_KEYWORDS = {
    "identity": [
        "passport", "id", "identity", "license", "licence", "ausweis", 
        "identitaet", "personalausweis", "reisepass", "driving"
    ],
    "residence_permit": [
        "residence", "permit", "aufenthalt", "bewilligung", "visa",
        "auslaenderausweis", "niederlassung"
    ],
    "proof_of_income": [
        "salary", "income", "payslip", "lohnausweis", "tax", "steuer",
        "gehalt", "einkommen", "lohn", "bank_statement", "kontoauszug",
        "account", "employment", "arbeitsvertrag"
    ],
    "credit_history": [
        "betreibung", "credit", "schulden", "debt", "kredit",
        "betreibungsauszug", "betreibungsregister"
    ],
    "property_valuation": [
        "valuation", "appraisal", "schaetzung", "bewertung", "listing",
        "immobilienbewertung", "gutachten", "property_value"
    ],
    "lex_koller_authorisation": [
        "lex_koller", "lexkoller", "koller", "authorisation", "authorization",
        "foreign_buyer", "auslaender_bewilligung"
    ],
    "land_registry_extract": [
        "land_registry", "grundbuch", "registry", "extract", "auszug",
        "grundbuchauszug", "cadastre", "kataster"
    ],
    "building_insurance": [
        "insurance", "versicherung", "gebaeudeversicherung", "building_insurance",
        "police", "policy", "hausrat"
    ],
    "purchase_contract": [
        "contract", "vertrag", "kaufvertrag", "reservation", "purchase",
        "agreement", "vereinbarung", "draft", "entwurf"
    ]
}


def classify_documents(
    documents: Annotated[str, "JSON string containing document URLs or filenames to classify. Format: {\"documents\": [\"file1.pdf\", \"file2.pdf\"]} or list of URLs"]
) -> Annotated[str, "JSON string with classification results: {\"classification\": {filename: label}}"]:
    """
    Classify mortgage documents based on their filenames.
    
    This is a file-routing assistant that maps each document to one of the
    predefined categories based only on the filename. For real OCR and page
    splitting, integrate with Azure Document Intelligence.
    
    Categories:
    - identity: ID, driver license or passport
    - residence_permit: Residence permit (if non-Swiss)
    - proof_of_income: Salary slips, tax return, account statements
    - credit_history: Betreibungsauszug (debt collection extract)
    - property_valuation: Listing, appraisal
    - lex_koller_authorisation: Lex Koller authorization
    - land_registry_extract: Land registry extract
    - building_insurance: Building insurance certificate or policy
    - purchase_contract: Draft purchase contract or reservation agreement
    - other: Unclassified documents
    
    Args:
        documents: JSON string with document list. Can be:
                  - {"documents": ["url1", "url2", ...]}
                  - ["url1", "url2", ...]
                  - {"doc_type": "url", ...}
    
    Returns:
        JSON string: {"classification": {<filename>: <label>, ...}}
    """
    tracing_manager = get_tracing_manager()
    
    try:
        with tracing_manager.trace_function_call(
            "classify_documents",
            parameters={"documents_input_length": len(documents)}
        ):
            # Parse the input
            try:
                data = json.loads(documents)
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "Invalid JSON format",
                    "classification": {}
                })
            
            # Extract document list from various input formats
            doc_list = []
            if isinstance(data, list):
                doc_list = data
            elif isinstance(data, dict):
                if "documents" in data:
                    doc_list = data["documents"]
                else:
                    # Assume dict of {type: url} - use the URLs
                    doc_list = list(data.values())
            
            # Classify each document
            classification = {}
            for doc_url in doc_list:
                if not doc_url or not isinstance(doc_url, str):
                    continue
                    
                # Extract filename from URL or path
                filename = _extract_filename(doc_url)
                
                # Classify based on filename
                label = _classify_by_filename(filename)
                classification[filename] = label
            
            logger.info(f"Classified {len(classification)} documents")
            
            return json.dumps({
                "classification": classification
            }, indent=2)
            
    except Exception as e:
        logger.error(f"Error classifying documents: {e}")
        if tracing_manager and tracing_manager.is_configured:
            with tracing_manager.trace_function_call(
                "classify_documents_error",
                parameters={"error": str(e)}
            ):
                pass
        return json.dumps({
            "error": f"Classification failed: {str(e)}",
            "classification": {}
        })


def _extract_filename(url_or_path: str) -> str:
    """Extract filename from URL or file path."""
    # Handle URLs
    if "://" in url_or_path:
        # Remove query parameters
        path = url_or_path.split("?")[0]
        # Get the last segment
        filename = path.rstrip("/").split("/")[-1]
    else:
        # Handle file paths
        filename = Path(url_or_path).name
    
    # URL decode if needed
    try:
        from urllib.parse import unquote
        filename = unquote(filename)
    except:
        pass
    
    return filename


def _classify_by_filename(filename: str) -> str:
    """Classify document based on filename using keyword matching."""
    # Normalize filename for matching
    normalized = filename.lower()
    normalized = re.sub(r'[_\-\s.]+', '_', normalized)  # Normalize separators
    
    # Check each category's keywords
    for category, keywords in CLASSIFICATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in normalized:
                return category
    
    # Default to "other" if no match
    return "other"


def get_classification_categories() -> Annotated[str, "JSON string listing all document classification categories"]:
    """
    Get the list of all available document classification categories.
    
    Returns:
        JSON string with category list and descriptions
    """
    categories_info = {
        "identity": "ID, driver license or passport",
        "residence_permit": "Residence permit (if non-Swiss)",
        "proof_of_income": "Salary slips, tax return, account statements",
        "credit_history": "Betreibungsauszug (debt collection extract)",
        "property_valuation": "Listing, appraisal",
        "lex_koller_authorisation": "Lex Koller authorization (for foreign buyers)",
        "land_registry_extract": "Land registry extract",
        "building_insurance": "Building insurance certificate or policy",
        "purchase_contract": "Draft purchase contract or reservation agreement",
        "other": "Unclassified documents"
    }
    
    return json.dumps({
        "categories": categories_info
    }, indent=2)


# Export functions for agent registration
classifier_functions: list[Callable[..., Any]] = [
    classify_documents,
    get_classification_categories
]

# Backwards compatibility alias
classify_mortgage_application = classify_documents
