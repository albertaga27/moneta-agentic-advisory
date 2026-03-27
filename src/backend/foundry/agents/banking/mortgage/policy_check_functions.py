"""
Policy check functions for Swiss mortgage regulatory compliance verification.
This module provides compliance checking against Swiss mortgage lending regulations.

Swiss Mortgage Rules Implemented:
1. Down-payment ≥ 20% of property value
2. Hard-equity ≥ 10% (own funds excluding Pillar 2/3 pledges)
3. Initial LTV ≤ 80% (loan cannot exceed 80% of property value)
4. Affordability ≤ 33% (annual housing costs / gross income)
5. Amortisation to ≤ 66.7% LTV within 15 years
6. Lex Koller compliance (foreign buyer restrictions)
7. Cross-border compliance (Swiss residence requirements)
8. Document completeness check
"""

import json
import logging
from typing import Annotated, Any, Callable
from pathlib import Path
from datetime import datetime

# Import tracing utilities from backend root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from tracing import get_tracing_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Swiss Mortgage Regulatory Limits
SWISS_MORTGAGE_LIMITS = {
    # Down payment and equity requirements
    "min_down_payment_percent": 20.0,      # Minimum down payment (%)
    "min_hard_equity_percent": 10.0,       # Minimum hard equity - own funds (%)
    "max_ltv_ratio": 80.0,                 # Maximum loan-to-value ratio (%)
    "target_ltv_15_years": 66.67,          # Target LTV after 15 years (%)
    
    # Affordability limits
    "max_affordability_ratio": 33.0,       # Maximum housing costs as % of income
    "stress_test_interest_rate": 5.0,      # Assumed interest rate for stress test (%)
    "annual_maintenance_rate": 1.0,        # Annual maintenance as % of property value
    
    # Amortisation requirements
    "amortisation_period_years": 15,       # Period to reach target LTV
    "first_mortgage_limit_percent": 65.0,  # First mortgage limit (% of property value)
    
    # Other limits
    "min_applicant_age": 18,
    "max_applicant_age_at_term_end": 70,
    "max_loan_term_years": 25,
}

# Required document categories for Swiss mortgages
REQUIRED_DOCUMENT_CATEGORIES = [
    "identity",
    "proof_of_income",
    "credit_history",
    "property_valuation",
]

# Optional but recommended documents
OPTIONAL_DOCUMENT_CATEGORIES = [
    "residence_permit",
    "lex_koller_authorisation",
    "land_registry_extract",
    "building_insurance",
    "purchase_contract",
]


def check_policy_compliance(
    application_data: Annotated[str, "JSON string containing the full mortgage application data including extracted document information, classification results, and form data"]
) -> Annotated[str, "JSON string containing policy compliance check results with rule_checks array"]:
    """
    Verify mortgage application against Swiss regulatory requirements and lending policies.
    
    Performs comprehensive compliance checks including:
    - Down-payment ≥ 20% of property value
    - Hard-equity ≥ 10% (excluding Pillar 2/3 pledges)  
    - Initial LTV ≤ 80% (loan-to-value ratio)
    - Affordability ≤ 33% of gross income
    - Amortisation to ≤ 66.7% LTV within 15 years
    - Lex Koller compliance (foreign buyer restrictions)
    - Cross-border compliance
    - Document completeness
    
    Args:
        application_data: JSON string with complete mortgage application data
                         including applicant info, property details, loan requirements,
                         classification results, and extracted document data
    
    Returns:
        JSON string containing:
        - approved: Boolean indicating if all mandatory checks passed
        - reasons: List of summary reasons
        - rule_checks: Array of detailed rule check results with status and details
    """
    tracing_manager = get_tracing_manager()
    
    try:
        with tracing_manager.trace_function_call(
            "check_policy_compliance",
            parameters={"application_data_length": len(application_data)}
        ):
            # Parse the application data
            try:
                data = json.loads(application_data)
            except json.JSONDecodeError:
                return json.dumps({
                    "approved": False,
                    "reasons": ["Invalid JSON format in application data"],
                    "rule_checks": [{
                        "rule": "Data validation",
                        "status": "fail",
                        "details": "Could not parse application data as JSON"
                    }]
                })
            
            # Extract application details from various possible structures
            app_data = data.get("data", data)
            documents = data.get("documents", {})
            classification = data.get("classification", {})
            extracted = data.get("extracted", {})
            
            # Initialize rule checks list
            rule_checks = []
            all_passed = True
            
            # === Extract key financial values ===
            property_value = float(app_data.get("property_value", 0))
            income = float(app_data.get("income", 0))
            liabilities = float(app_data.get("liabilities", 0))
            down_payment_pc = float(app_data.get("down_payment_pc", 0))
            
            # Pillar pledges (soft equity)
            pledge_p2 = app_data.get("pledge_p2", False)
            pledge_p2_amount = float(app_data.get("pledge_p2_amount", 0))
            pledge_p3 = app_data.get("pledge_p3", False)
            pledge_p3_amount = float(app_data.get("pledge_p3_amount", 0))
            
            # Citizenship/residency
            swiss_or_eu_citizen = app_data.get("swiss_or_eu_citizen", True)
            
            # Calculate derived values
            down_payment_amount = (down_payment_pc / 100) * property_value if property_value else 0
            loan_amount = property_value - down_payment_amount if property_value else 0
            total_pledge_amount = (pledge_p2_amount if pledge_p2 else 0) + (pledge_p3_amount if pledge_p3 else 0)
            hard_equity_amount = down_payment_amount - total_pledge_amount
            hard_equity_pc = (hard_equity_amount / property_value * 100) if property_value else 0
            ltv = (loan_amount / property_value * 100) if property_value else 0
            
            # === RULE 1: Down-payment ≥ 20% ===
            min_down_payment = SWISS_MORTGAGE_LIMITS["min_down_payment_percent"]
            if down_payment_pc >= min_down_payment:
                rule_checks.append({
                    "rule": "Down-payment ≥ 20 %",
                    "status": "pass",
                    "details": f"down payment = {down_payment_pc}% (CHF {down_payment_amount:,.0f}) meets minimum requirement of ≥{min_down_payment:.0f}%"
                })
            else:
                all_passed = False
                rule_checks.append({
                    "rule": "Down-payment ≥ 20 %",
                    "status": "fail",
                    "details": f"down payment = {down_payment_pc}% (CHF {down_payment_amount:,.0f}) is below minimum requirement of ≥{min_down_payment:.0f}%"
                })
            
            # === RULE 2: Hard-equity ≥ 10% ===
            min_hard_equity = SWISS_MORTGAGE_LIMITS["min_hard_equity_percent"]
            if hard_equity_pc >= min_hard_equity:
                rule_checks.append({
                    "rule": "Hard-equity ≥ 10 %",
                    "status": "pass",
                    "details": f"hard equity = {hard_equity_pc:.1f}% (CHF {hard_equity_amount:,.0f}) exceeds minimum requirement of ≥{min_hard_equity:.0f}%"
                })
            else:
                all_passed = False
                pledge_note = ""
                if total_pledge_amount > 0:
                    pledge_note = f" (Pillar pledges of CHF {total_pledge_amount:,.0f} do not count as hard equity)"
                rule_checks.append({
                    "rule": "Hard-equity ≥ 10 %",
                    "status": "fail",
                    "details": f"hard equity = {hard_equity_pc:.1f}% (CHF {hard_equity_amount:,.0f}) is below minimum requirement of ≥{min_hard_equity:.0f}%{pledge_note}"
                })
            
            # === RULE 3: Initial LTV ≤ 80% ===
            max_ltv = SWISS_MORTGAGE_LIMITS["max_ltv_ratio"]
            if ltv <= max_ltv:
                rule_checks.append({
                    "rule": "Initial LTV ≤ policy limit",
                    "status": "pass",
                    "details": f"loan amount = CHF{loan_amount:,.0f} yields LTV = {ltv:.0f}% (≤{max_ltv:.0f}% required)"
                })
            else:
                all_passed = False
                rule_checks.append({
                    "rule": "Initial LTV ≤ policy limit",
                    "status": "fail",
                    "details": f"loan amount = CHF{loan_amount:,.0f} yields LTV = {ltv:.1f}% (exceeds {max_ltv:.0f}% maximum)"
                })
            
            # === RULE 4: Affordability ≤ 33% ===
            max_affordability = SWISS_MORTGAGE_LIMITS["max_affordability_ratio"]
            stress_rate = SWISS_MORTGAGE_LIMITS["stress_test_interest_rate"]
            maintenance_rate = SWISS_MORTGAGE_LIMITS["annual_maintenance_rate"]
            
            if income > 0 and loan_amount > 0:
                # Calculate annual costs at stress rate
                annual_interest = loan_amount * (stress_rate / 100)
                annual_maintenance = property_value * (maintenance_rate / 100)
                
                # Amortisation: second mortgage portion to be paid off over 15 years
                first_mortgage_limit = property_value * (SWISS_MORTGAGE_LIMITS["first_mortgage_limit_percent"] / 100)
                second_mortgage = max(0, loan_amount - first_mortgage_limit)
                annual_amortisation = second_mortgage / SWISS_MORTGAGE_LIMITS["amortisation_period_years"] if second_mortgage > 0 else 0
                
                total_annual_costs = annual_interest + annual_maintenance + annual_amortisation
                affordability_ratio = (total_annual_costs / income) * 100
                
                if affordability_ratio <= max_affordability:
                    rule_checks.append({
                        "rule": "Affordability ≤ 33 %",
                        "status": "pass",
                        "details": f"annual stress costs = CHF{annual_interest:,.0f} (interest) + CHF{annual_maintenance:,.0f} (maintenance) + CHF{annual_amortisation:,.0f} (amortisation) ≈ CHF{total_annual_costs:,.0f}; ratio = {total_annual_costs:,.0f}/{income:,.0f}×100 ≈ {affordability_ratio:.1f}% (≤{max_affordability:.0f}%)"
                    })
                else:
                    all_passed = False
                    rule_checks.append({
                        "rule": "Affordability ≤ 33 %",
                        "status": "fail",
                        "details": f"annual stress costs = CHF{annual_interest:,.0f} (interest) + CHF{annual_maintenance:,.0f} (maintenance) + CHF{annual_amortisation:,.0f} (amortisation) ≈ CHF{total_annual_costs:,.0f}; ratio = {total_annual_costs:,.0f}/{income:,.0f}×100 ≈ {affordability_ratio:.1f}% (exceeds {max_affordability:.0f}% limit)"
                    })
            else:
                all_passed = False
                rule_checks.append({
                    "rule": "Affordability ≤ 33 %",
                    "status": "fail",
                    "details": "unable to calculate affordability - missing income or loan amount"
                })
            
            # === RULE 5: Amortisation to ≤ 66.7% in 15 years ===
            target_ltv_15y = SWISS_MORTGAGE_LIMITS["target_ltv_15_years"]
            first_mortgage_limit = property_value * (SWISS_MORTGAGE_LIMITS["first_mortgage_limit_percent"] / 100)
            target_loan_15y = property_value * (target_ltv_15y / 100)
            
            if property_value > 0:
                second_mortgage = max(0, loan_amount - first_mortgage_limit)
                final_loan_after_15y = loan_amount - second_mortgage  # After paying off 2nd mortgage
                final_ltv_15y = (final_loan_after_15y / property_value) * 100
                
                if final_ltv_15y <= target_ltv_15y:
                    rule_checks.append({
                        "rule": "Amortisation to ≤ 66.7 % in 15 y",
                        "status": "pass",
                        "details": f"second mortgage = CHF{loan_amount:,.0f} - CHF{first_mortgage_limit:,.0f} = CHF{second_mortgage:,.0f}; amortised over 15 years results in final loan ≈ CHF{final_loan_after_15y:,.0f} (<{target_ltv_15y:.1f}% of property value)"
                    })
                else:
                    all_passed = False
                    rule_checks.append({
                        "rule": "Amortisation to ≤ 66.7 % in 15 y",
                        "status": "fail",
                        "details": f"final LTV after 15 years = {final_ltv_15y:.1f}% exceeds target of {target_ltv_15y:.1f}%"
                    })
            else:
                rule_checks.append({
                    "rule": "Amortisation to ≤ 66.7 % in 15 y",
                    "status": "fail",
                    "details": "unable to calculate amortisation - missing property value"
                })
            
            # === RULE 6: Lex Koller compliance ===
            # Lex Koller restricts property purchases by non-Swiss/non-EU citizens
            has_identity = _has_document_category(classification, "identity")
            has_lex_koller_auth = _has_document_category(classification, "lex_koller_authorisation")
            
            if swiss_or_eu_citizen:
                rule_checks.append({
                    "rule": "Lex Koller compliance",
                    "status": "pass",
                    "details": "applicant is Swiss/EU citizen; Lex Koller restrictions not applicable" if has_identity else "ch id provided; applicant assumed swiss so lex koller not applicable"
                })
            else:
                # Non-Swiss/EU citizen needs Lex Koller authorization
                if has_lex_koller_auth:
                    rule_checks.append({
                        "rule": "Lex Koller compliance",
                        "status": "pass",
                        "details": "non-Swiss/EU applicant has provided Lex Koller authorization"
                    })
                else:
                    all_passed = False
                    rule_checks.append({
                        "rule": "Lex Koller compliance",
                        "status": "fail",
                        "details": "non-Swiss/EU applicant requires Lex Koller authorization to purchase property in Switzerland"
                    })
            
            # === RULE 7: Cross-border compliance ===
            has_residence_permit = _has_document_category(classification, "residence_permit")
            
            if swiss_or_eu_citizen:
                rule_checks.append({
                    "rule": "Cross-border compliance",
                    "status": "pass",
                    "details": "applicant appears domestic; no cross-border restrictions applicable"
                })
            elif has_residence_permit:
                rule_checks.append({
                    "rule": "Cross-border compliance",
                    "status": "pass",
                    "details": "non-Swiss applicant has provided residence permit; cross-border requirements met"
                })
            else:
                # For non-Swiss without residence permit, add a warning but don't fail
                rule_checks.append({
                    "rule": "Cross-border compliance",
                    "status": "warning",
                    "details": "non-Swiss applicant without residence permit on file; may require additional documentation for cross-border mortgage"
                })
            
            # === RULE 8: Document completeness ===
            doc_completeness = {}
            missing_required = []
            
            for required_cat in REQUIRED_DOCUMENT_CATEGORIES:
                if _has_document_category(classification, required_cat):
                    doc_completeness[required_cat] = "present"
                else:
                    doc_completeness[required_cat] = "missing"
                    missing_required.append(required_cat)
            
            if not missing_required:
                rule_checks.append({
                    "rule": "Document completeness",
                    "status": "pass",
                    "details": doc_completeness
                })
            else:
                all_passed = False
                rule_checks.append({
                    "rule": "Document completeness",
                    "status": "fail",
                    "details": doc_completeness
                })
            
            # === BUILD FINAL DECISION ===
            if all_passed:
                reasons = ["All mandatory rules passed!"]
            else:
                failed_rules = [rc["rule"] for rc in rule_checks if rc["status"] == "fail"]
                reasons = [f"Failed checks: {', '.join(failed_rules)}"]
            
            # Add warnings to reasons if any
            warning_rules = [rc for rc in rule_checks if rc.get("status") == "warning"]
            if warning_rules:
                reasons.append(f"Warnings: {', '.join([w['rule'] for w in warning_rules])}")
            
            logger.info(
                f"Policy check complete: approved={all_passed}, "
                f"passed={sum(1 for rc in rule_checks if rc['status'] == 'pass')}, "
                f"failed={sum(1 for rc in rule_checks if rc['status'] == 'fail')}"
            )
            
            return json.dumps({
                "approved": all_passed,
                "reasons": reasons,
                "rule_checks": rule_checks
            }, indent=2)
            
    except Exception as e:
        logger.error(f"Error checking policy compliance: {e}")
        if tracing_manager and tracing_manager.is_configured:
            with tracing_manager.trace_function_call(
                "check_policy_compliance_error",
                parameters={"error": str(e)}
            ):
                pass
        return json.dumps({
            "approved": False,
            "reasons": [f"Policy check failed: {str(e)}"],
            "rule_checks": [{
                "rule": "System error",
                "status": "fail",
                "details": f"An error occurred during policy check: {str(e)}"
            }]
        })


def _has_document_category(classification: dict, category: str) -> bool:
    """
    Check if the classification contains a document of the given category.
    
    Args:
        classification: Dict mapping filenames to category labels
        category: The category to check for
        
    Returns:
        True if any document is classified as the given category
    """
    if not classification:
        return False
    
    # Handle both direct dict and nested structure
    if isinstance(classification, dict):
        # Check if it's the classification dict itself
        for filename, label in classification.items():
            if isinstance(label, str) and label.lower() == category.lower():
                return True
    
    return False


def get_swiss_mortgage_limits() -> Annotated[str, "JSON string with Swiss mortgage regulatory limits"]:
    """
    Get the Swiss mortgage regulatory limits used for policy checks.
    
    Returns:
        JSON string containing all regulatory limits and their descriptions
    """
    limits_info = {
        "down_payment": {
            "minimum_percent": SWISS_MORTGAGE_LIMITS["min_down_payment_percent"],
            "description": "Minimum down payment as percentage of property value"
        },
        "hard_equity": {
            "minimum_percent": SWISS_MORTGAGE_LIMITS["min_hard_equity_percent"],
            "description": "Minimum hard equity (own funds excluding Pillar 2/3 pledges)"
        },
        "ltv": {
            "maximum_percent": SWISS_MORTGAGE_LIMITS["max_ltv_ratio"],
            "description": "Maximum initial loan-to-value ratio"
        },
        "affordability": {
            "maximum_percent": SWISS_MORTGAGE_LIMITS["max_affordability_ratio"],
            "stress_test_rate": SWISS_MORTGAGE_LIMITS["stress_test_interest_rate"],
            "description": "Maximum housing costs as percentage of gross income (calculated at stress rate)"
        },
        "amortisation": {
            "target_ltv_percent": SWISS_MORTGAGE_LIMITS["target_ltv_15_years"],
            "period_years": SWISS_MORTGAGE_LIMITS["amortisation_period_years"],
            "description": "Second mortgage must be amortised to reach target LTV within period"
        },
        "required_documents": REQUIRED_DOCUMENT_CATEGORIES,
        "optional_documents": OPTIONAL_DOCUMENT_CATEGORIES
    }
    
    return json.dumps(limits_info, indent=2)


# Export functions for agent registration
policy_check_functions: list[Callable[..., Any]] = [
    check_policy_compliance,
    get_swiss_mortgage_limits
]
