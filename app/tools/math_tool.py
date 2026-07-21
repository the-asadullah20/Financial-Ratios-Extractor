"""
Math Tools - Exact Python financial math calculator for calculating financial ratios.
Prevents LLM arithmetic calculation errors.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("math_tool")


def calculate_ratio_value(numerator: float, denominator: float, precision: int = 4) -> Optional[float]:
    """Calculates numerator / denominator safely rounded to specified precision."""
    if denominator == 0 or denominator is None or numerator is None:
        return None
    try:
        return round(float(numerator) / float(denominator), precision)
    except (ZeroDivisionError, ValueError, TypeError):
        return None


def calculate_financial_ratios(statement_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Computes all standard financial ratios from extracted statement data for every period/year found.
    
    Expected statement_data schema:
    {
      "Current Assets": {"2025": 23362.0, "2024": 25382.0},
      "Current Liabilities": {"2025": 10566.0, ...},
      ...
    }
    """
    ratios_output: Dict[str, Dict[str, Any]] = {}

    # Extract set of all periods (years/dates)
    all_periods = set()
    for field_values in statement_data.values():
        if isinstance(field_values, dict):
            all_periods.update(field_values.keys())

    # Helper to get numeric value for a field and period
    def val(field_name: str, period: str) -> Optional[float]:
        data = statement_data.get(field_name) or {}
        if isinstance(data, dict):
            return data.get(period)
        return None

    # 1. Current Ratio
    curr_ratio: Dict[str, Any] = {"source": "Calculated (Current Assets / Current Liabilities)"}
    for p in all_periods:
        ca = val("Current Assets", p)
        cl = val("Current Liabilities", p)
        if ca is not None and cl is not None:
            curr_ratio[p] = calculate_ratio_value(ca, cl)
        else:
            curr_ratio[p] = None
    ratios_output["Current Ratio"] = curr_ratio

    # 2. Debt Ratio
    debt_ratio: Dict[str, Any] = {"source": "Calculated (Total Liabilities / Total Assets)"}
    for p in all_periods:
        tl = val("Total Liabilities", p)
        ta = val("Total Assets", p)
        if tl is not None and ta is not None:
            debt_ratio[p] = calculate_ratio_value(tl, ta)
        else:
            debt_ratio[p] = None
    ratios_output["Debt Ratio"] = debt_ratio

    # 3. Gross Margin
    gross_margin: Dict[str, Any] = {"source": "Calculated (Gross Profit / Revenue)"}
    for p in all_periods:
        gp = val("Gross Profit", p)
        rev = val("Revenue", p)
        if gp is not None and rev is not None:
            gross_margin[p] = calculate_ratio_value(gp, rev)
        else:
            gross_margin[p] = None
    ratios_output["Gross Margin"] = gross_margin

    # 4. Net Profit Margin
    net_margin: Dict[str, Any] = {"source": "Calculated (Net Income / Revenue)"}
    for p in all_periods:
        ni = val("Net Income", p)
        rev = val("Revenue", p)
        if ni is not None and rev is not None:
            net_margin[p] = calculate_ratio_value(ni, rev)
        else:
            net_margin[p] = None
    ratios_output["Net Profit Margin"] = net_margin

    # 5. Debt-to-Equity (if Total Equity available)
    de_ratio: Dict[str, Any] = {"source": "Calculated (Total Liabilities / Total Equity)"}
    for p in all_periods:
        tl = val("Total Liabilities", p)
        eq = val("Total Equity", p)
        if tl is not None and eq is not None:
            de_ratio[p] = calculate_ratio_value(tl, eq)
        else:
            de_ratio[p] = None
    ratios_output["Debt-to-Equity"] = de_ratio

    # 6. Loan-to-Income (LTI)
    lti_ratio: Dict[str, Any] = {"source": "Calculated (Long Term Debt / Revenue)"}
    for p in all_periods:
        ltd = val("Long Term Debt", p)
        rev = val("Revenue", p)
        if ltd is not None and rev is not None:
            lti_ratio[p] = calculate_ratio_value(ltd, rev)
        else:
            lti_ratio[p] = None
    ratios_output["Loan-to-Income (LTI)"] = lti_ratio

    # 7. Debt-to-Income (DTI)
    dti_ratio: Dict[str, Any] = {"source": "Calculated (Long Term Debt / Net Income)"}
    for p in all_periods:
        ltd = val("Long Term Debt", p)
        ni = val("Net Income", p)
        if ltd is not None and ni is not None:
            dti_ratio[p] = calculate_ratio_value(ltd, ni)
        else:
            dti_ratio[p] = None
    ratios_output["Debt-to-Income (DTI)"] = dti_ratio

    return ratios_output
