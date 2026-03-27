"""
Document extraction functions using Azure Document Intelligence.
This module provides document analysis and data extraction for mortgage documents.
When Azure Document Intelligence is not configured, uses PyPDF2/pypdf to extract text from local PDF files.
"""

import os
import json
import logging
import base64
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


class DocumentExtractionFunctions:
    """
    Document Extraction Functions using Azure Document Intelligence.
    Extracts structured data from mortgage-related documents.
    """
    
    def __init__(self):
        """Initialize the Document Extraction Functions with Azure Document Intelligence client."""
        self.document_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.document_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        self._client = None
        
        if not self.document_intelligence_endpoint:
            logger.warning(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT not set. "
                "Document extraction will use mock implementation."
            )
    
    def _get_client(self):
        """Lazily initialize the Document Intelligence client."""
        if self._client is None and self.document_intelligence_endpoint:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
                from azure.identity import DefaultAzureCredential
                
                if self.document_intelligence_key:
                    self._client = DocumentIntelligenceClient(
                        endpoint=self.document_intelligence_endpoint,
                        credential=AzureKeyCredential(self.document_intelligence_key)
                    )
                else:
                    self._client = DocumentIntelligenceClient(
                        endpoint=self.document_intelligence_endpoint,
                        credential=DefaultAzureCredential()
                    )
                logger.info("Initialized Azure Document Intelligence client")
            except ImportError:
                logger.warning(
                    "azure-ai-documentintelligence package not installed. "
                    "Using mock implementation."
                )
            except Exception as e:
                logger.error(f"Failed to initialize Document Intelligence client: {e}")
        
        return self._client
    
    def extract_document_data(
        self,
        document_url: Annotated[str, "URL or path to the document to extract data from"],
        document_type: Annotated[str, "Type of document: id_document, income_proof, bank_statement, property_valuation"]
    ) -> Annotated[str, "JSON string containing extracted document data"]:
        """
        Extract structured data from a mortgage-related document using Azure Document Intelligence.
        
        Supports the following document types:
        - id_document: Identity documents (passport, driver's license, ID card)
        - income_proof: Payslips, employment letters, tax returns
        - bank_statement: Bank statements showing income and expenses
        - property_valuation: Property valuation reports
        
        Args:
            document_url: URL or file path to the document
            document_type: Type of document to process
            
        Returns:
            JSON string containing extracted fields, confidence scores, and any validation issues
        """
        tracing_manager = get_tracing_manager()
        
        try:
            with tracing_manager.trace_function_call(
                "extract_document_data",
                parameters={
                    "document_url": document_url,
                    "document_type": document_type
                }
            ):
                client = self._get_client()
                
                if client:
                    # Use Azure Document Intelligence for real extraction
                    return self._extract_with_azure_di(client, document_url, document_type)
                else:
                    # Use mock implementation for development/testing
                    return self._mock_extraction(document_url, document_type)
                    
        except Exception as e:
            logger.error(f"Error extracting document data: {e}")
            if tracing_manager and tracing_manager.is_configured:
                with tracing_manager.trace_function_call(
                    "extract_document_data_error",
                    parameters={"error": str(e), "document_type": document_type}
                ):
                    pass
            return json.dumps({
                "error": f"Document extraction failed: {str(e)}",
                "document_type": document_type,
                "extracted_fields": {}
            })
    
    def _extract_with_azure_di(
        self, 
        client, 
        document_url: str, 
        document_type: str
    ) -> str:
        """Extract data using Azure Document Intelligence."""
        try:
            # Determine the model to use based on document type
            model_id = self._get_model_for_document_type(document_type)
            
            # Analyze the document
            if document_url.startswith("http"):
                poller = client.begin_analyze_document(
                    model_id=model_id,
                    analyze_request={"url_source": document_url}
                )
            else:
                # Handle local file path
                with open(document_url, "rb") as f:
                    poller = client.begin_analyze_document(
                        model_id=model_id,
                        analyze_request=f,
                        content_type="application/octet-stream"
                    )
            
            result = poller.result()
            
            # Extract fields from result
            extracted_fields = {}
            confidence_scores = []
            
            for document in result.documents:
                for field_name, field_value in document.fields.items():
                    if field_value:
                        extracted_fields[field_name] = {
                            "value": field_value.content if hasattr(field_value, 'content') else str(field_value.value),
                            "confidence": field_value.confidence if hasattr(field_value, 'confidence') else 1.0
                        }
                        if hasattr(field_value, 'confidence'):
                            confidence_scores.append(field_value.confidence)
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Get raw text content
            raw_text = result.content if hasattr(result, 'content') else ""
            
            return json.dumps({
                "status": "success",
                "document_type": document_type,
                "extracted_fields": extracted_fields,
                "confidence_score": round(overall_confidence, 3),
                "raw_text": raw_text[:1000],  # Limit raw text length
                "validation_errors": []
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Azure DI extraction error: {e}")
            return json.dumps({
                "error": f"Azure Document Intelligence extraction failed: {str(e)}",
                "document_type": document_type,
                "extracted_fields": {}
            })
    
    def _get_model_for_document_type(self, document_type: str) -> str:
        """Get the appropriate Azure DI model for the document type."""
        model_mapping = {
            "id_document": "prebuilt-idDocument",
            "income_proof": "prebuilt-invoice",  # Or custom model
            "bank_statement": "prebuilt-document",
            "property_valuation": "prebuilt-document"
        }
        return model_mapping.get(document_type, "prebuilt-document")
    
    def _get_mock_data_folder(self) -> Path:
        """Get the path to the mock mortgage data folder."""
        # Navigate from mortgage -> banking -> agents -> foundry -> backend -> src -> data/mortgage
        return Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "mortgage"
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("pypdf not installed. Trying PyPDF2...")
            try:
                from PyPDF2 import PdfReader
                
                reader = PdfReader(str(file_path))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                return "\n".join(text_parts)
            except ImportError:
                logger.error("Neither pypdf nor PyPDF2 is installed. Cannot read PDF files.")
                return ""
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def _extract_text_from_image(self, file_path: Path) -> str:
        """Extract text from an image file using OCR if available, otherwise return metadata."""
        try:
            # Try using pytesseract for OCR if available
            from PIL import Image
            import pytesseract
            
            image = Image.open(str(file_path))
            text = pytesseract.image_to_string(image)
            return text
        except ImportError:
            logger.info(f"OCR not available for image {file_path.name}. Returning file metadata.")
            return f"[Image file: {file_path.name}. OCR not available - install pytesseract for text extraction]"
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return f"[Image file: {file_path.name}. Error during processing: {e}]"
    
    def _detect_profile_from_url(self, document_url: str) -> str | None:
        """Detect the profile name (pete or bradley) from the document URL or content."""
        url_lower = document_url.lower()
        if 'pete' in url_lower:
            return 'pete'
        elif 'bradley' in url_lower or 'bardely' in url_lower:
            return 'bradley'
        return None
    
    def _find_matching_file(self, document_url: str, document_type: str) -> Path | None:
        """Find a matching file in the mock data folder based on document URL and type."""
        profile = self._detect_profile_from_url(document_url)
        if not profile:
            # Try to detect from document type keywords
            logger.warning(f"Could not detect profile from URL: {document_url}")
            return None
        
        mock_folder = self._get_mock_data_folder() / profile
        if not mock_folder.exists():
            logger.warning(f"Mock data folder not found: {mock_folder}")
            return None
        
        # Try to find a matching file based on the document URL filename
        # Extract filename from URL if it's a data URL
        if "base64," in document_url:
            # For base64 data URLs, we can't determine the original filename
            # Try to match based on document type
            return self._find_file_by_type(mock_folder, document_type)
        
        # Try to extract filename from URL path
        url_filename = document_url.split("/")[-1].split("?")[0].lower()
        
        # Look for a matching file in the folder
        for file_path in mock_folder.iterdir():
            if file_path.is_file():
                if file_path.name.lower() == url_filename:
                    return file_path
                # Fuzzy match on filename parts
                if url_filename.replace("_", " ").replace("-", " ") in file_path.name.lower().replace("_", " ").replace("-", " "):
                    return file_path
        
        # Fallback to document type matching
        return self._find_file_by_type(mock_folder, document_type)
    
    def _find_file_by_type(self, folder: Path, document_type: str) -> Path | None:
        """Find a file in the folder that matches the document type."""
        type_keywords = {
            "identity": ["id", "passport", "license", "driver"],
            "id_document": ["id", "passport", "license", "driver"],
            "residence_permit": ["residence", "permit", "residency"],
            "proof_of_income": ["salary", "income", "pay", "statement"],
            "income_proof": ["salary", "income", "pay", "statement"],
            "credit_history": ["debt", "credit", "betreibung"],
            "property_valuation": ["apartment", "property", "valuation", "buy", "description"],
            "bank_statement": ["bank", "statement", "account"],
            "building_insurance": ["insurance", "building"],
            "purchase_contract": ["contract", "purchase", "reservation"],
            "land_registry_extract": ["registry", "grundbuch"],
            "lex_koller_authorisation": ["lex", "koller", "authorization"]
        }
        
        keywords = type_keywords.get(document_type, [])
        
        for file_path in folder.iterdir():
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                for keyword in keywords:
                    if keyword in filename_lower:
                        return file_path
        
        # Return first PDF file as fallback
        for file_path in folder.iterdir():
            if file_path.suffix.lower() == '.pdf':
                return file_path
        
        return None
    
    def _extract_fields_from_text(self, text: str, document_type: str) -> dict:
        """Extract structured fields from raw text based on document type."""
        fields = {}
        
        # Common patterns to extract
        patterns = {
            "name": r"(?:Name|Full Name|Applicant)[:\s]+([A-Za-z\s]+?)(?:\n|$)",
            "date": r"(?:Date|Issued)[:\s]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
            "amount": r"(?:CHF|USD|EUR)\s*([\d',]+\.?\d*)",
            "address": r"(?:Address|Location)[:\s]+(.+?)(?:\n|$)",
            "salary": r"(?:Salary|Income|Gross)[:\s]*(?:CHF)?\s*([\d',]+)",
            "employer": r"(?:Employer|Company)[:\s]+(.+?)(?:\n|$)",
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = {
                    "value": match.group(1).strip(),
                    "confidence": 0.85
                }
        
        # Add document-type specific extraction
        if document_type in ["income_proof", "proof_of_income"]:
            # Look for salary amounts
            salary_matches = re.findall(r"(?:Net|Gross|Salary|Monthly)[^0-9]*(\d{1,3}(?:[',]\d{3})*(?:\.\d{2})?)", text, re.IGNORECASE)
            if salary_matches:
                fields["salary_amount"] = {"value": salary_matches[0], "confidence": 0.82}
        
        elif document_type in ["property_valuation"]:
            # Look for property values
            value_matches = re.findall(r"(?:Price|Value|CHF)[^0-9]*(\d{1,3}(?:[',]\d{3})*)", text, re.IGNORECASE)
            if value_matches:
                fields["property_value"] = {"value": value_matches[0], "confidence": 0.80}
        
        return fields
    
    def _mock_extraction(self, document_url: str, document_type: str) -> str:
        """
        Extract data from local PDF/image files when Azure DI is not configured.
        Uses PyPDF2/pypdf to read PDF files and pytesseract for images if available.
        """
        logger.info(f"Using local file extraction for {document_type}")
        
        # Try to find the matching file
        file_path = self._find_matching_file(document_url, document_type)
        
        if file_path and file_path.exists():
            logger.info(f"Found matching file: {file_path}")
            
            # Extract text based on file type
            suffix = file_path.suffix.lower()
            if suffix == '.pdf':
                raw_text = self._extract_text_from_pdf(file_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                raw_text = self._extract_text_from_image(file_path)
            else:
                raw_text = f"[Unsupported file type: {suffix}]"
            
            if raw_text:
                # Extract structured fields from text
                extracted_fields = self._extract_fields_from_text(raw_text, document_type)
                
                # Add source file info
                extracted_fields["source_file"] = {
                    "value": file_path.name,
                    "confidence": 1.0
                }
                
                # Calculate overall confidence
                confidences = [f["confidence"] for f in extracted_fields.values() if isinstance(f, dict)]
                overall_confidence = sum(confidences) / len(confidences) if confidences else 0.75
                
                return json.dumps({
                    "status": "success",
                    "document_type": document_type,
                    "extracted_fields": extracted_fields,
                    "confidence_score": round(overall_confidence, 3),
                    "raw_text": raw_text[:2000],  # Include more raw text for context
                    "validation_errors": [],
                    "note": f"Extracted from local file: {file_path.name}"
                }, indent=2)
        
        # Fallback to mock data if file not found
        logger.warning(f"Could not find matching file for {document_type}, using fallback mock data")
        return self._fallback_mock_extraction(document_url, document_type)
    
    def _fallback_mock_extraction(self, document_url: str, document_type: str) -> str:
        """Fallback mock extraction when no file can be found."""
        mock_data = {
            "id_document": {
                "full_name": {"value": "John Smith", "confidence": 0.95},
                "date_of_birth": {"value": "1985-03-15", "confidence": 0.92},
                "document_number": {"value": "AB123456", "confidence": 0.98},
                "expiry_date": {"value": "2028-03-14", "confidence": 0.94},
                "nationality": {"value": "UK", "confidence": 0.97}
            },
            "income_proof": {
                "employer_name": {"value": "Acme Corporation", "confidence": 0.93},
                "gross_salary": {"value": "75000", "confidence": 0.89},
                "pay_period": {"value": "monthly", "confidence": 0.95},
                "employment_start_date": {"value": "2019-06-01", "confidence": 0.88},
                "position": {"value": "Senior Engineer", "confidence": 0.91}
            },
            "bank_statement": {
                "account_holder": {"value": "John Smith", "confidence": 0.96},
                "account_number": {"value": "****5678", "confidence": 0.99},
                "statement_period": {"value": "2024-01-01 to 2024-01-31", "confidence": 0.94},
                "closing_balance": {"value": "15234.56", "confidence": 0.92},
                "average_monthly_income": {"value": "6250.00", "confidence": 0.85}
            },
            "property_valuation": {
                "property_address": {"value": "123 Main Street, London, SW1A 1AA", "confidence": 0.97},
                "estimated_value": {"value": "450000", "confidence": 0.88},
                "property_type": {"value": "Semi-detached house", "confidence": 0.95},
                "valuation_date": {"value": "2024-01-15", "confidence": 0.99},
                "condition": {"value": "Good", "confidence": 0.86}
            }
        }
        
        extracted_fields = mock_data.get(document_type, {
            "content": {"value": "Generic document content", "confidence": 0.80}
        })
        
        # Calculate mock confidence
        confidences = [f["confidence"] for f in extracted_fields.values()]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return json.dumps({
            "status": "success",
            "document_type": document_type,
            "extracted_fields": extracted_fields,
            "confidence_score": round(overall_confidence, 3),
            "raw_text": f"[Fallback mock data for {document_type}]",
            "validation_errors": [],
            "note": "Fallback mock data - local file not found and Azure DI not configured"
        }, indent=2)


# Global instance
_document_extraction_functions = None


def get_document_extraction_functions() -> DocumentExtractionFunctions:
    """Get or create the global document extraction functions instance."""
    global _document_extraction_functions
    if _document_extraction_functions is None:
        _document_extraction_functions = DocumentExtractionFunctions()
    return _document_extraction_functions


def extract_document_data(
    document_url: Annotated[str, "URL or path to the document"],
    document_type: Annotated[str, "Type of document: id_document, income_proof, bank_statement, property_valuation"]
) -> str:
    """Wrapper function for agent execution."""
    return get_document_extraction_functions().extract_document_data(document_url, document_type)


# Export functions for agent registration
document_extraction_functions: list[Callable[..., Any]] = [
    extract_document_data
]
