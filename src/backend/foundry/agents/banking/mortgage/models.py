"""
Mortgage Models

Data models for mortgage application processing.
"""

from dataclasses import dataclass, field
from datetime import datetime as dt
from typing import Any


@dataclass
class MortgageRequest:
    """
    Represents a mortgage application request.
    
    Attributes:
        request_id: Unique identifier for the mortgage request
        data: Dictionary containing applicant and mortgage details
        documents: Dictionary mapping document filenames to document URLs
        status: Current status of the request (RECEIVED, CLASSIFYING, EXTRACTING, 
                POLICY_CHECK, PENDING_REVIEW, APPROVED, REJECTED, COMPLETED)
        created_utc: ISO format timestamp of creation
        updated_utc: ISO format timestamp of last update
        error: Error message if processing failed
        classification: Classification results mapping filename to document category
        extracted: Extracted data from each document, keyed by filename
        decision: Final decision with approved flag, reasons list, and rule_checks
    """
    request_id: str
    data: dict[str, Any]
    documents: dict[str, str]
    status: str = "RECEIVED"
    created_utc: str = field(default_factory=lambda: dt.utcnow().isoformat())
    updated_utc: str = field(default_factory=lambda: dt.utcnow().isoformat())
    error: str | None = None
    classification: dict[str, str] | None = None
    extracted: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the MortgageRequest to a dictionary."""
        return {
            "request_id": self.request_id,
            "data": self.data,
            "documents": self.documents,
            "status": self.status,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "error": self.error,
            "classification": self.classification,
            "extracted": self.extracted,
            "decision": self.decision
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MortgageRequest":
        """Create a MortgageRequest from a dictionary."""
        return cls(
            request_id=data["request_id"],
            data=data["data"],
            documents=data["documents"],
            status=data.get("status", "RECEIVED"),
            created_utc=data.get("created_utc", dt.utcnow().isoformat()),
            updated_utc=data.get("updated_utc", dt.utcnow().isoformat()),
            error=data.get("error"),
            classification=data.get("classification"),
            extracted=data.get("extracted"),
            decision=data.get("decision")
        )
    
    def update_status(self, new_status: str) -> None:
        """Update the status and timestamp."""
        self.status = new_status
        self.updated_utc = dt.utcnow().isoformat()
    
    def set_error(self, error_message: str) -> None:
        """Set error and update status."""
        self.error = error_message
        self.status = "ERROR"
        self.updated_utc = dt.utcnow().isoformat()


@dataclass
class DocumentExtractionResult:
    """
    Result of document extraction from Azure Document Intelligence.
    
    Attributes:
        document_type: Type of document (ID, income_proof, property_valuation, etc.)
        extracted_fields: Dictionary of extracted field names and values
        confidence_score: Overall confidence score of extraction
        raw_text: Raw extracted text from the document
        validation_errors: List of any validation issues found
    """
    document_type: str
    extracted_fields: dict[str, Any]
    confidence_score: float
    raw_text: str
    validation_errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_type": self.document_type,
            "extracted_fields": self.extracted_fields,
            "confidence_score": self.confidence_score,
            "raw_text": self.raw_text,
            "validation_errors": self.validation_errors
        }


@dataclass
class PolicyCheckResult:
    """
    Result of policy compliance check.
    
    Attributes:
        is_compliant: Whether the application meets all regulatory requirements
        checks_passed: List of passed compliance checks
        checks_failed: List of failed compliance checks
        warnings: List of non-blocking warnings
        recommendation: Overall recommendation (APPROVE, REJECT, NEEDS_REVIEW)
        reasoning: Detailed reasoning for the decision
    """
    is_compliant: bool
    checks_passed: list[str]
    checks_failed: list[str]
    warnings: list[str]
    recommendation: str
    reasoning: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_compliant": self.is_compliant,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning
        }
