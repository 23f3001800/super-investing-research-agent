"""Agent tools: search_pack, read_source, get_source_metadata.

These are ordinary deterministic functions.  The research agent LLM
chooses which one to call; the tool itself does not make autonomous
decisions about the final thesis.
"""

from typing import Dict, List, Optional

from source_registry import SourceRegistry


def search_pack(
    registry: SourceRegistry,
    query: str,
    filters: Optional[Dict] = None,
) -> str:
    """Search the research-pack documents by keyword.

    Returns a formatted string suitable for direct inclusion in the
    LLM's observation context.
    """
    results = registry.search(query, filters=filters)
    if not results:
        return f"[search_pack] No results for query: '{query}'"

    lines = [f"[search_pack] {len(results)} result(s) for '{query}':\n"]
    for r in results:
        lines.append(
            f"  {r['source_id']} | {r['source_name']} | score={r['score']}\n"
            f"    Snippet: {r['relevance_snippet']}\n"
        )
    return "\n".join(lines)


def read_source(
    registry: SourceRegistry,
    source_id: str,
    passage: Optional[str] = None,
) -> str:
    """Read a source document or locate a specific passage within it.

    If *passage* is given, returns ~500 chars around the first match
    (case-insensitive) with the match highlighted in **bold**.
    Otherwise returns the full body (truncated to 2 000 chars).
    """
    doc = registry.get(source_id)
    if doc is None:
        return f"[read_source] Unknown source ID: {source_id}"

    header = (
        f"[read_source] {doc.source_id} — {doc.source_name}\n"
        f"  Type: {doc.source_type} | Published: {doc.published}\n"
        f"  URL:  {doc.url}\n"
        f"{'─' * 60}\n"
    )

    if passage:
        idx = doc.body.lower().find(passage.lower())
        if idx == -1:
            return header + f"  Passage '{passage}' not found in this document."
        # Extract context window
        start = max(0, idx - 250)
        end = min(len(doc.body), idx + len(passage) + 250)
        before = doc.body[start:idx]
        match = doc.body[idx : idx + len(passage)]
        after = doc.body[idx + len(passage) : end]
        excerpt = f"{'…' if start > 0 else ''}{before}**{match}**{after}{'…' if end < len(doc.body) else ''}"
        return header + excerpt
    else:
        body = doc.body[:2000]
        if len(doc.body) > 2000:
            body += "\n… [truncated]"
        return header + body


def get_source_metadata(registry: SourceRegistry, source_id: str) -> str:
    """Return metadata for a single source document."""
    doc = registry.get(source_id)
    if doc is None:
        return f"[get_source_metadata] Unknown source ID: {source_id}"

    return (
        f"[get_source_metadata] {doc.source_id}\n"
        f"  Source:    {doc.source_name}\n"
        f"  Type:      {doc.source_type}\n"
        f"  Published: {doc.published}\n"
        f"  URL:       {doc.url}\n"
        f"  File:      {doc.filename}\n"
        f"  Hash:      {doc.file_hash[:16]}…\n"
    )
