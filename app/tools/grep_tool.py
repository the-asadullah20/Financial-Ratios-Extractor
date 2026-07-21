"""
Mirage Grep Tool - Exact keyword & regex text search across document Markdown.
"""
import re
from typing import List, Dict, Any


def grep_search_markdown(query: str, markdown_text: str, is_regex: bool = False, max_results: int = 15) -> List[Dict[str, Any]]:
    """
    Searches raw document Markdown for exact query matches or regex patterns.
    Returns matching lines with page numbers and surrounding context.
    """
    matches = []
    lines = markdown_text.splitlines()
    current_page = 1

    pattern = query if is_regex else re.escape(query)

    for idx, line in enumerate(lines, start=1):
        # Update current page tracking
        page_match = re.search(r"--- PAGE (\d+) ---", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if re.search(pattern, line, re.IGNORECASE):
            # Extract surrounding context (1 line before and 1 line after)
            context_start = max(0, idx - 2)
            context_end = min(len(lines), idx + 1)
            snippet = "\n".join(lines[context_start:context_end])

            matches.append({
                "line_number": idx,
                "page_number": current_page,
                "matched_line": line.strip(),
                "context": snippet,
            })
            if len(matches) >= max_results:
                break

    return matches
