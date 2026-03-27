"""
Query functions for mortgage request lookup and filtering.

This module provides functions to help users find and filter their mortgage requests
by request ID, status, property value, or other criteria.
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level storage for the persister instance (set by orchestrator)
_persister = None
_current_user_id = None


def set_query_context(persister, user_id: str):
    """
    Set the query context for mortgage request lookups.
    Called by the orchestrator before running the workflow.
    
    Args:
        persister: CosmosPersister instance
        user_id: Current user ID
    """
    global _persister, _current_user_id
    _persister = persister
    _current_user_id = user_id


def clear_query_context():
    """Clear the query context after workflow completion."""
    global _persister, _current_user_id
    _persister = None
    _current_user_id = None


def list_mortgage_requests(
    status_filter: Annotated[Optional[str], "Optional status filter: RECEIVED, PROCESSING, PENDING_REVIEW, APPROVED, REJECTED, ERROR"] = None,
    limit: Annotated[int, "Maximum number of requests to return (default 10)"] = 10
) -> str:
    """
    List all mortgage requests for the current user with optional filtering.
    
    Returns a summary of requests including request ID, status, property value, and creation date.
    """
    try:
        if not _persister or not _current_user_id:
            return json.dumps({
                "error": "Query context not initialized. Please try again.",
                "requests": []
            })
        
        # Load user document
        user_doc = _persister.load_user_doc(_current_user_id)
        requests = user_doc.get("requests", [])
        
        # Apply status filter if provided
        if status_filter:
            status_upper = status_filter.upper()
            requests = [r for r in requests if r.get("status", "").upper() == status_upper]
        
        # Limit results
        requests = requests[-limit:]  # Get most recent
        
        # Build summary list
        summary = []
        for r in requests:
            summary.append({
                "request_id": r.get("request_id"),
                "status": r.get("status", "UNKNOWN"),
                "property_value": r.get("data", {}).get("property_value"),
                "created_utc": r.get("created_utc"),
                "decision": r.get("decision", {}).get("recommendation") if r.get("decision") else None
            })
        
        result = {
            "total_requests": len(user_doc.get("requests", [])),
            "filtered_count": len(summary),
            "requests": summary
        }
        
        logger.info(f"Listed {len(summary)} mortgage requests for user {_current_user_id}")
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error listing mortgage requests: {e}")
        return json.dumps({"error": str(e), "requests": []})


def get_request_by_id(
    request_id: Annotated[str, "The mortgage request ID to look up (e.g., 'MR-2024-001' or a UUID)"]
) -> str:
    """
    Get detailed information about a specific mortgage request by its ID.
    
    Returns the full request details including application data, documents, status, and decision.
    """
    try:
        if not _persister or not _current_user_id:
            return json.dumps({
                "error": "Query context not initialized. Please try again.",
                "request": None
            })
        
        # Load user document
        user_doc = _persister.load_user_doc(_current_user_id)
        requests = user_doc.get("requests", [])
        
        # Search for the request
        for r in requests:
            if r.get("request_id") == request_id:
                logger.info(f"Found request {request_id} for user {_current_user_id}")
                return json.dumps({
                    "found": True,
                    "request": r
                }, indent=2, default=str)
        
        # Also try partial match (case-insensitive)
        request_id_lower = request_id.lower()
        for r in requests:
            if request_id_lower in r.get("request_id", "").lower():
                logger.info(f"Found request via partial match: {r.get('request_id')}")
                return json.dumps({
                    "found": True,
                    "partial_match": True,
                    "request": r
                }, indent=2, default=str)
        
        return json.dumps({
            "found": False,
            "error": f"No request found with ID '{request_id}'",
            "suggestion": "Use list_mortgage_requests to see all available requests."
        })
        
    except Exception as e:
        logger.error(f"Error getting request by ID: {e}")
        return json.dumps({"error": str(e), "request": None})


def search_requests(
    search_term: Annotated[str, "Search term to find in request data (property value, status, dates, etc.)"]
) -> str:
    """
    Search mortgage requests by various criteria.
    
    Searches across request IDs, status, property values, and dates.
    Useful when the user mentions partial information about their request.
    """
    try:
        if not _persister or not _current_user_id:
            return json.dumps({
                "error": "Query context not initialized. Please try again.",
                "matches": []
            })
        
        # Load user document
        user_doc = _persister.load_user_doc(_current_user_id)
        requests = user_doc.get("requests", [])
        
        search_lower = search_term.lower()
        matches = []
        
        for r in requests:
            # Search in various fields
            searchable_text = " ".join([
                str(r.get("request_id", "")),
                str(r.get("status", "")),
                str(r.get("data", {}).get("property_value", "")),
                str(r.get("created_utc", "")),
                str(r.get("decision", {}).get("recommendation", "") if r.get("decision") else ""),
            ])
            
            if search_lower in searchable_text.lower():
                matches.append({
                    "request_id": r.get("request_id"),
                    "status": r.get("status", "UNKNOWN"),
                    "property_value": r.get("data", {}).get("property_value"),
                    "created_utc": r.get("created_utc"),
                    "match_reason": "Found in request data"
                })
        
        result = {
            "search_term": search_term,
            "matches_found": len(matches),
            "matches": matches
        }
        
        logger.info(f"Search found {len(matches)} matches for '{search_term}'")
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error searching requests: {e}")
        return json.dumps({"error": str(e), "matches": []})


def get_latest_request() -> str:
    """
    Get the most recent mortgage request for the current user.
    
    Useful when the user refers to "my request" or "my application" without specifying an ID.
    """
    try:
        if not _persister or not _current_user_id:
            return json.dumps({
                "error": "Query context not initialized. Please try again.",
                "request": None
            })
        
        # Load user document
        user_doc = _persister.load_user_doc(_current_user_id)
        requests = user_doc.get("requests", [])
        
        if not requests:
            return json.dumps({
                "found": False,
                "message": "You don't have any mortgage requests yet.",
                "request": None
            })
        
        # Get the most recent request
        latest = requests[-1]
        
        logger.info(f"Retrieved latest request: {latest.get('request_id')}")
        
        return json.dumps({
            "found": True,
            "request": latest
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error getting latest request: {e}")
        return json.dumps({"error": str(e), "request": None})


def get_requests_by_status(
    status: Annotated[str, "The status to filter by: RECEIVED, PROCESSING, PENDING_REVIEW, APPROVED, REJECTED, ERROR"]
) -> str:
    """
    Get all mortgage requests with a specific status.
    
    Useful when user asks about pending requests, approved applications, etc.
    """
    try:
        if not _persister or not _current_user_id:
            return json.dumps({
                "error": "Query context not initialized. Please try again.",
                "requests": []
            })
        
        status_upper = status.upper()
        
        # Load user document
        user_doc = _persister.load_user_doc(_current_user_id)
        requests = user_doc.get("requests", [])
        
        # Filter by status
        filtered = [
            {
                "request_id": r.get("request_id"),
                "status": r.get("status"),
                "property_value": r.get("data", {}).get("property_value"),
                "created_utc": r.get("created_utc"),
                "updated_utc": r.get("updated_utc"),
                "decision": r.get("decision", {}).get("recommendation") if r.get("decision") else None
            }
            for r in requests
            if r.get("status", "").upper() == status_upper
        ]
        
        result = {
            "status_filter": status_upper,
            "count": len(filtered),
            "requests": filtered
        }
        
        logger.info(f"Found {len(filtered)} requests with status {status_upper}")
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error getting requests by status: {e}")
        return json.dumps({"error": str(e), "requests": []})


# Export functions as a list for use with agent tools
query_functions = [
    list_mortgage_requests,
    get_request_by_id,
    search_requests,
    get_latest_request,
    get_requests_by_status,
]
