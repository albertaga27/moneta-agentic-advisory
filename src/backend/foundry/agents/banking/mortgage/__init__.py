"""
Mortgage Agent Package

This package contains the mortgage processing agents using Microsoft Agent Framework.
It provides functionality to process mortgage applications through a sequential workflow:

1. Classifier Agent - Classifies mortgage documents based on filenames
2. Document Extraction Agent - Extracts data from submitted documents using Azure Document Intelligence
3. Policy Check Agent - Verifies mortgage application against regulatory requirements
4. Mortgage Orchestrator - Sequential orchestration with Human-in-the-Loop

Modules:
    models: Data models for mortgage processing
    classifier_agent: Mortgage document classifier (filename-based routing)
    document_extraction_agent: Azure Document Intelligence integration
    policy_check_agent: Regulatory compliance verification
    mortgage_orchestrator: Sequential workflow orchestrator
"""

from .models import MortgageRequest
from .classifier_functions import (
    classifier_functions,
    classify_documents,
    classify_mortgage_application,  # Backwards compatibility alias
    get_classification_categories,
    DOCUMENT_CATEGORIES
)
from .document_extraction_functions import (
    document_extraction_functions,
    extract_document_data
)
from .policy_check_functions import policy_check_functions, check_policy_compliance

__all__ = [
    'MortgageRequest',
    'classifier_functions',
    'classify_documents',
    'classify_mortgage_application',
    'get_classification_categories',
    'DOCUMENT_CATEGORIES',
    'document_extraction_functions',
    'extract_document_data',
    'policy_check_functions',
    'check_policy_compliance'
]
