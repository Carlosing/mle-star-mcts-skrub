"""Web search utility for MLE-STAR using DuckDuckGo.

This module exposes a tiny API around DuckDuckGo text search. It is intentionally
provider-specific because the project decided to fix the provider as DuckDuckGo.
"""

from typing import Any

from ddgs import DDGS


def search_web(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Search DuckDuckGo and return a list of result dicts.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return. Defaults to 5.

    Returns:
        A list of dictionaries with keys such as "title", "href", and "body".
        Returns an empty list if the search fails for any reason.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return results
    except Exception:
        return []


def format_results_for_prompt(results: list[dict[str, Any]]) -> str:
    """Convert search results into a string suitable for an LLM prompt.

    Args:
        results: List of search result dictionaries.

    Returns:
        A formatted string with numbered results, or a neutral message if the
        list is empty.
    """
    if not results:
        return "No web search results available."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        body = item.get("body", "").replace("\n", " ")
        lines.append(f"{i}. {title} - {body}")
    return "\n".join(lines)
