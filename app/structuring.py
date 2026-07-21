"""
LLM Structuring Engine module.
Takes unstructured agent ratio analysis and formats it into canonical structured JSON.
"""
import logging
from typing import Dict, Any

import google.generativeai as genai
from app.config import settings
from app.schema import build_gemini_instructions, normalize_output
from app.utils import extract_json_object

logger = logging.getLogger("structuring")


def structure_financial_ratios(extracted_statement: Dict[str, Any], computed_ratios: Dict[str, Any], unstructured_output: str) -> Dict[str, Any]:
    """
    Combines agent output, extracted line items, and computed math ratios into the canonical JSON structure.
    """
    structured = {
        "company_name": extracted_statement.get("company_name") or "Unknown Company",
        "ticker": extracted_statement.get("ticker"),
        "country": extracted_statement.get("country") or "US",
        "currency": extracted_statement.get("currency") or "USD",
        "statement_data": extracted_statement.get("statement_data") or {},
        "ratios": computed_ratios,
        "notes": unstructured_output
    }

    # Normalize through schema safety checks
    normalized = normalize_output(structured)
    return normalized
