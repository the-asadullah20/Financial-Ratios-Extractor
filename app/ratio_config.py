"""
Configuration module defining financial ratios to calculate, their formulas,
and required statement line items.
"""
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class RatioDefinition(BaseModel):
    name: str
    description: str
    formula: str
    required_fields: List[str]
    category: str = "general"


# Default target ratios configuration
DEFAULT_RATIOS: Dict[str, RatioDefinition] = {
    "Current Ratio": RatioDefinition(
        name="Current Ratio",
        description="Measures liquidity by comparing current assets to current liabilities.",
        formula="Current Assets / Current Liabilities",
        required_fields=["Current Assets", "Current Liabilities"],
        category="liquidity",
    ),
    "Debt Ratio": RatioDefinition(
        name="Debt Ratio",
        description="Measures financial leverage by comparing total liabilities to total assets.",
        formula="Total Liabilities / Total Assets",
        required_fields=["Total Liabilities", "Total Assets"],
        category="leverage",
    ),
    "Gross Margin": RatioDefinition(
        name="Gross Margin",
        description="Measures profitability as percentage of revenue remaining after cost of goods.",
        formula="Gross Profit / Revenue",
        required_fields=["Gross Profit", "Revenue"],
        category="profitability",
    ),
    "Net Profit Margin": RatioDefinition(
        name="Net Profit Margin",
        description="Measures net profit generated per dollar of revenue.",
        formula="Net Income / Revenue",
        required_fields=["Net Income", "Revenue"],
        category="profitability",
    ),
    "Debt-to-Equity": RatioDefinition(
        name="Debt-to-Equity",
        description="Measures relative proportion of shareholder equity and debt used to finance assets.",
        formula="Total Liabilities / Total Equity",
        required_fields=["Total Liabilities", "Total Equity"],
        category="leverage",
    ),
    "EBITDA Margin": RatioDefinition(
        name="EBITDA Margin",
        description="Measures earnings before interest, taxes, depreciation, and amortization relative to revenue.",
        formula="(Gross Profit - Operating Expenses + Depreciation) / Revenue",
        required_fields=["Revenue", "Gross Profit"],
        category="profitability",
    ),
    "Loan-to-Income (LTI)": RatioDefinition(
        name="Loan-to-Income (LTI)",
        description="Measures total long-term debt relative to annual revenue.",
        formula="Long Term Debt / Revenue",
        required_fields=["Long Term Debt", "Revenue"],
        category="debt_coverage",
    ),
    "Debt-to-Income (DTI)": RatioDefinition(
        name="Debt-to-Income (DTI)",
        description="Measures total debt obligations relative to annual net income.",
        formula="Long Term Debt / Net Income",
        required_fields=["Long Term Debt", "Net Income"],
        category="debt_coverage",
    ),
}


def get_active_ratio_configs() -> Dict[str, Any]:
    """Returns the dictionary of all configured financial ratio definitions."""
    return {k: v.model_dump() for k, v in DEFAULT_RATIOS.items()}
