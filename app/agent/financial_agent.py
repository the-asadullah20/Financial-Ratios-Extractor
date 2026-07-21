"""
LangGraph Deep AI Agent for Financial Ratio Extraction.
Orchestrates tool execution (Grep Search, Vector Retrieval, Math Tools)
using a ReAct Agent Graph to analyze financial documents.
"""
import logging
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.config import settings
from app.ratio_config import get_active_ratio_configs
from app.tools.grep_tool import grep_search_markdown
from app.tools.retrieval_tool import semantic_retrieval_search
from app.tools.math_tool import calculate_financial_ratios

logger = logging.getLogger("financial_agent")


class AgentState(TypedDict):
    document_id: str
    markdown_content: str
    grep_results: List[Dict[str, Any]]
    retrieval_chunks: List[Dict[str, Any]]
    extracted_statement: Dict[str, Any]
    computed_ratios: Dict[str, Any]
    unstructured_output: str


def node_grep_search(state: AgentState) -> AgentState:
    """Executes keyword grep search over markdown content to locate financial tables."""
    logger.info("Agent: Running Mirage Grep tool search across full document...")
    markdown = state["markdown_content"]
    
    target_terms = [
        "Current Assets", "Current Liabilities", "Total Assets",
        "Total Liabilities", "Revenue", "Gross Profit", "Net Income",
        "Long Term Debt", "Total Equity", "Balance Sheet", "Income Statement"
    ]
    
    all_matches = []
    for term in target_terms:
        matches = grep_search_markdown(term, markdown, max_results=10)
        all_matches.extend(matches)

    state["grep_results"] = all_matches
    return state


def node_semantic_retrieval(state: AgentState) -> AgentState:
    """Executes vector retrieval search against Qdrant index."""
    logger.info("Agent: Running Semantic Retrieval tool search...")
    doc_id = state.get("document_id")
    queries = [
        "Consolidated balance sheet current assets current liabilities total debt",
        "Consolidated statement of income net revenue gross profit net income",
        "Financial ratio notes long term debt obligations and equity"
    ]
    
    chunks = []
    for q in queries:
        res = semantic_retrieval_search(q, document_id=doc_id, top_k=6)
        chunks.extend(res)

    state["retrieval_chunks"] = chunks
    return state


def node_llm_reasoning(state: AgentState) -> AgentState:
    """Uses Gemini LLM to interpret retrieved contexts and extract structured figures across ALL pages."""
    logger.info("Agent: Processing evidence across all pages with Gemini LLM...")
    import google.generativeai as genai
    import json
    from app.utils import extract_json_object

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

    grep_text = "\n".join([f"Page {m['page_number']}: {m['matched_line']}" for m in state.get("grep_results", [])[:30]])
    retrieval_text = "\n".join([f"Chunk (P.{c.get('page_number',1)}): {c.get('text','')}" for c in state.get("retrieval_chunks", [])[:15]])

    prompt = f"""
You are an expert financial agent.
Examine the following extracted document evidence (from keyword grep search and semantic vector retrieval) and the FULL document text:

=== GREP EVIDENCE ===
{grep_text}

=== VECTOR RETRIEVAL EVIDENCE ===
{retrieval_text}

=== FULL UNTRUNCATED DOCUMENT CONTEXT ===
{state["markdown_content"]}

Your task:
1. Identify company name, ticker (if known), country, currency.
2. Extract numeric values for these exact statement items across all available periods/years from ALL pages:
   - Current Assets
   - Current Liabilities
   - Total Assets
   - Total Liabilities
   - Revenue
   - Gross Profit
   - Net Income
   - Total Equity (optional/when available)
   - Long Term Debt (optional/when available)
   - Cash And Equivalents (optional/when available)

Return ONLY a valid JSON object of the form:
{{
  "company_name": "...",
  "ticker": "...",
  "country": "...",
  "currency": "...",
  "statement_data": {{
    "Current Assets": {{"2025": 100, "2024": 90}},
    "Current Liabilities": {{"2025": 50, "2024": 45}},
    "Total Assets": {{"2025": 200, "2024": 180}},
    "Total Liabilities": {{"2025": 100, "2024": 95}},
    "Revenue": {{"2025": 300, "2024": 280}},
    "Gross Profit": {{"2025": 150, "2024": 140}},
    "Net Income": {{"2025": 40, "2024": 35}}
  }}
}}
"""
    try:
        response = model.generate_content(prompt)
        parsed = extract_json_object(response.text)
        state["extracted_statement"] = parsed
    except Exception as exc:
        logger.warning("LLM reasoning extraction warning (%s). Using fallback parser.", exc)
        state["extracted_statement"] = {
            "company_name": "Target Company",
            "ticker": None,
            "country": "US",
            "currency": "USD",
            "statement_data": {}
        }
    
    return state


def node_math_calculator(state: AgentState) -> AgentState:
    """Uses Python Math Tool to compute exact financial ratios deterministically."""
    logger.info("Agent: Running Python Math tool to compute financial ratios...")
    extracted = state.get("extracted_statement") or {}
    statement_data = extracted.get("statement_data") or {}

    ratios = calculate_financial_ratios(statement_data)
    state["computed_ratios"] = ratios

    # Produce unstructured narrative summary
    summary = [f"Financial Ratio Analysis for {extracted.get('company_name', 'Company')}:"]
    for ratio_name, period_vals in ratios.items():
        vals_str = ", ".join([f"{k}: {v}" for k, v in period_vals.items() if k != "source" and v is not None])
        summary.append(f"- {ratio_name}: {vals_str} ({period_vals.get('source', '')})")
    
    state["unstructured_output"] = "\n".join(summary)
    return state


def build_financial_agent_graph() -> StateGraph:
    """Builds the LangGraph agent state graph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("grep_search", node_grep_search)
    workflow.add_node("semantic_retrieval", node_semantic_retrieval)
    workflow.add_node("llm_reasoning", node_llm_reasoning)
    workflow.add_node("math_calculator", node_math_calculator)

    workflow.set_entry_point("grep_search")
    workflow.add_edge("grep_search", "semantic_retrieval")
    workflow.add_edge("semantic_retrieval", "llm_reasoning")
    workflow.add_edge("llm_reasoning", "math_calculator")
    workflow.add_edge("math_calculator", END)

    return workflow.compile()


financial_agent_app = build_financial_agent_graph()


def run_financial_agent(document_id: str, markdown_content: str) -> Dict[str, Any]:
    """Runs the LangGraph financial AI agent end-to-end."""
    initial_state: AgentState = {
        "document_id": document_id,
        "markdown_content": markdown_content,
        "grep_results": [],
        "retrieval_chunks": [],
        "extracted_statement": {},
        "computed_ratios": {},
        "unstructured_output": ""
    }

    final_state = financial_agent_app.invoke(initial_state)
    return final_state
