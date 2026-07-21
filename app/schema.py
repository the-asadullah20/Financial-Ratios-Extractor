"""
Canonical target schema for extracted financial statement JSON.

This mirrors the format found across the sample files you provided
(Nike.json, Walmart.json, Barclays.json, etc). Companies report line items
under slightly different names (e.g. "Revenue" vs "Net Revenue" vs
"Total Revenues"), so Gemini is instructed to normalize everything into
this canonical set of keys.
"""

# Core line items that MUST be present in statement_data (normalized names).
CORE_STATEMENT_FIELDS = [
    "Current Assets",
    "Current Liabilities",
    "Total Assets",
    "Total Liabilities",
    "Revenue",
    "Gross Profit",
    "Net Income",
]

# Extra line items to include *when available* in the source document
# (seen in the more detailed sample files: Nike, Walmart, Cocacola, Pepsico).
OPTIONAL_STATEMENT_FIELDS = [
    "Total Equity",
    "Cash And Equivalents",
    "Inventories",
    "Long Term Debt",
]

# Ratios that must always be computed / included.
REQUIRED_RATIOS = [
    "Current Ratio",   # Current Assets / Current Liabilities
    "Debt Ratio",       # Total Liabilities / Total Assets
    "Gross Margin",     # Gross Profit / Revenue
    "Net Profit Margin",  # Net Income / Revenue
]

TOP_LEVEL_REQUIRED = [
    "company_name",
    "ticker",
    "country",
    "currency",
    "statement_data",
    "ratios",
]

TOP_LEVEL_OPTIONAL = [
    "fiscal_year_end",
    "source_document",
    "notes",
]

EXAMPLE_JSON = {
    "company_name": "Nike, Inc.",
    "ticker": "NKE",
    "country": "US",
    "currency": "USD",
    "fiscal_year_end": "May 31",
    "source_document": "Nike 10-K (fiscal year ended May 31, 2025)",
    "statement_data": {
        "Current Assets": {"2025-05-31": 23362.0, "2024-05-31": 25382.0},
        "Total Assets": {"2025-05-31": 36579.0, "2024-05-31": 38110.0},
        "Current Liabilities": {"2025-05-31": 10566.0, "2024-05-31": 10593.0},
        "Total Liabilities": {"2025-05-31": 23366.0, "2024-05-31": 23680.0},
        "Total Equity": {"2025-05-31": 13213.0, "2024-05-31": 14430.0},
        "Revenue": {"2025-05-31": 46309.0, "2024-05-31": 51362.0},
        "Gross Profit": {"2025-05-31": 19790.0, "2024-05-31": 22887.0},
        "Net Income": {"2025-05-31": 3219.0, "2024-05-31": 5700.0},
        "Cash And Equivalents": {"2025-05-31": 7464.0, "2024-05-31": 9860.0},
        "Inventories": {"2025-05-31": 7489.0, "2024-05-31": 7519.0},
        "Long Term Debt": {"2025-05-31": 7961.0, "2024-05-31": 7903.0},
    },
    "ratios": {
        "Current Ratio": {
            "2025": 2.2111, "2024": 2.3961,
            "source": "Calculated (Current Assets / Current Liabilities)",
        },
        "Debt Ratio": {
            "2025": 0.6388, "2024": 0.6214,
            "source": "Calculated (Total Liabilities / Total Assets)",
        },
        "Gross Margin": {
            "2025": 0.4273, "2024": 0.4456,
            "source": "Calculated (Gross Profit / Revenue)",
        },
        "Net Profit Margin": {
            "2025": 0.0695, "2024": 0.1110,
            "source": "Calculated (Net Income / Revenue)",
        },
    },
    "notes": "All figures in USD millions except ratios.",
}


def normalize_output(data: dict) -> dict:
    """
    Safety net around whatever Gemini returns: makes sure every required
    top-level key and every required statement_data/ratios key is present
    (filled with None if genuinely missing), without dropping any extra
    fields the model included.
    """
    data = dict(data)  # shallow copy

    for key in TOP_LEVEL_REQUIRED:
        data.setdefault(key, None)

    statement_data = data.get("statement_data") or {}
    for key in CORE_STATEMENT_FIELDS:
        statement_data.setdefault(key, None)
    data["statement_data"] = statement_data

    ratios = data.get("ratios") or {}
    for key in REQUIRED_RATIOS:
        ratios.setdefault(key, None)
    data["ratios"] = ratios

    return data


def build_gemini_instructions() -> str:
    """Returns the instruction block describing the required output schema."""
    import json

    return f"""
You are a financial-statement data extraction engine.

You will be given raw OCR text extracted from a company's annual report /
10-K / financial statement PDF (page by page, in order). Your job is to
read it and output ONE JSON object that captures the company's balance
sheet and income statement figures, normalized into the exact schema below.

REQUIRED TOP-LEVEL KEYS: {TOP_LEVEL_REQUIRED}
OPTIONAL TOP-LEVEL KEYS (include if you can determine them, omit otherwise): {TOP_LEVEL_OPTIONAL}

"statement_data" MUST contain these normalized keys (use these exact names,
regardless of what the source document calls them):
{CORE_STATEMENT_FIELDS}

Also include these OPTIONAL keys in "statement_data" whenever the source
document reports them:
{OPTIONAL_STATEMENT_FIELDS}

Field name mapping guidance (source wording -> canonical key):
- "Total Current Assets" / "Current Assets" -> "Current Assets"
- "Total Current Liabilities" / "Current Liabilities" -> "Current Liabilities"
- "Total Assets" -> "Total Assets"
- "Total Liabilities" -> "Total Liabilities"
- "Total Shareholders Equity" / "Total Equity" / "Stockholders Equity" -> "Total Equity"
- "Revenues" / "Net Revenue" / "Total Revenues" / "Net Operating Revenues" -> "Revenue"
- "Cost of Sales" / "Cost of Goods Sold" -> keep as context, but always compute "Gross Profit" = Revenue - Cost of Sales if Gross Profit isn't stated directly
- "Net Income Attributable To ..." / "Consolidated Net Income Attributable To ..." -> "Net Income"
- "Cash and Cash Equivalents" / "Cash And Equivalents" -> "Cash And Equivalents"
- "Long Term Debt Obligations" / "Long Term Debt" -> "Long Term Debt"

Each statement_data value is an object mapping a period/date (prefer
ISO format "YYYY-MM-DD" if a specific period end date is stated, otherwise
a 4-digit year string) to a numeric value. Include every period found in
the document (there will usually be at least 2 - current year and prior
year comparison).

"ratios" MUST contain these exact keys: {REQUIRED_RATIOS}
Each ratio value is an object mapping year (4-digit string) to a numeric
value, PLUS a "source" string describing how it was derived, e.g.
"Calculated from 10-K (Total Current Assets / Total Current Liabilities)".
Compute ratios yourself from the statement_data values you extracted:
- Current Ratio = Current Assets / Current Liabilities
- Debt Ratio = Total Liabilities / Total Assets
- Gross Margin = Gross Profit / Revenue
- Net Profit Margin = Net Income / Revenue
Round ratio values to 4 decimal places.

Rules:
- "company_name" must be the full legal company name found in the document.
- "ticker" is the stock ticker symbol if stated or well known, else your best inference, else null.
- "country" is the ISO-2 country code of headquarters if determinable (e.g. "US", "GB"), else your best inference.
- "currency" is the ISO currency code used in the statements (e.g. "USD", "GBP").
- All monetary figures should be numbers (not strings), in the same unit the
  document reports (usually millions) - do not rescale.
- If a required field truly cannot be found anywhere in the text, set its
  value to null rather than guessing a number.
- Add a "notes" field with any caveats about the extraction (unit basis,
  assumptions made, fields not found, etc).
- Output ONLY the JSON object. No markdown fences, no commentary, no preamble.

EXAMPLE of the exact shape expected (values are illustrative only):
{json.dumps(EXAMPLE_JSON, indent=2)}
""".strip()
