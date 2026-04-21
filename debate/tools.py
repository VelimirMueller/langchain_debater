"""Tavily search + extract, wrapped as LangChain tools for debater use.

Exposed as LangChain Tool instances so they trace cleanly in LangSmith
and Langfuse under their canonical names (tavily_search, tavily_extract).

This module is the one place to tune retrieval knobs (max_results,
search_depth, extract_depth) without touching orchestration code.
"""

from langchain_tavily import TavilyExtract, TavilySearch


def get_research_tools() -> list:
    """Return [search, extract] with debate-appropriate defaults.

    max_results=4 keeps each search response to roughly 1.5k input tokens.
    search_depth='basic' is 1 credit per call; 'advanced' is 2 credits
    and rarely worth it for debate-scale queries.
    include_answer=False — we want raw sources the LLM cites, not
    Tavily's pre-synthesised one-liner.
    """
    search = TavilySearch(
        max_results=4,
        search_depth="basic",
        include_answer=False,
    )
    extract = TavilyExtract(extract_depth="basic")
    return [search, extract]
