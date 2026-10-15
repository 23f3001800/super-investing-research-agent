
"""Source registry: loads documents, assigns stable IDs, and provides search."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SourceDocument:
    """A single source document from the research pack."""
    source_id: str          # e.g. "S1"
    filename: str           # e.g. "businessdaily_2026-08-09.md"
    source_name: str        # from frontmatter 'source' field
    url: str                # from frontmatter 'url' field
    published: str          # from frontmatter 'published' field
    source_type: str        # from frontmatter 'type' field
    body: str               # everything after the second '---'
    file_hash: str          # sha256 of raw file contents


@dataclass
class SourceRegistry:
    """Registry of all source documents with search capabilities."""
    documents: Dict[str, SourceDocument] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(cls, pack_dir: str | Path) -> "SourceRegistry":
        """Load all .md files from *pack_dir*, parse metadata, assign IDs."""
        registry = cls()
        pack_path = Path(pack_dir)
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack directory not found: {pack_path}")

        md_files = sorted(pack_path.glob("*.md"))
        for idx, path in enumerate(md_files, start=1):
            raw = path.read_text(encoding="utf-8")
            file_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            meta, body = _parse_frontmatter(raw)
            doc = SourceDocument(
                source_id=f"S{idx}",
                filename=path.name,
                source_name=meta.get("source", ""),
                url=meta.get("url", ""),
                published=meta.get("published", ""),
                source_type=meta.get("type", ""),
                body=body,
                file_hash=file_hash,
            )
            registry.documents[doc.source_id] = doc
        return registry

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, source_id: str) -> Optional[SourceDocument]:
        """Get a document by its source ID (e.g. 'S1')."""
        return self.documents.get(source_id)

    def list_sources(self) -> List[Dict]:
        """Return summary metadata for every source."""
        return [
            {
                "source_id": d.source_id,
                "filename": d.filename,
                "source_name": d.source_name,
                "source_type": d.source_type,
                "published": d.published,
                "url": d.url,
            }
            for d in self.documents.values()
        ]

    def validate_citation_ids(self, citation_ids: set) -> tuple:
        """Return (valid_ids, unknown_ids)."""
        known = set(self.documents.keys())
        valid = citation_ids & known
        unknown = citation_ids - known
        return valid, unknown

    # ------------------------------------------------------------------
    # Keyword search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 5,
    ) -> List[Dict]:
        """Simple keyword search across document bodies.

        Returns list of dicts with keys:
            source_id, filename, source_name, relevance_snippet, score

        Filters can include:
            source_type   – exact match
            published_after  – YYYY-MM-DD string
            published_before – YYYY-MM-DD string
        """
        keywords = query.lower().split()
        if not keywords:
            return []

        filters = filters or {}
        results: List[Dict] = []

        for doc in self.documents.values():
            # ---- apply filters ----
            if "source_type" in filters:
                if filters["source_type"].lower() not in doc.source_type.lower():
                    continue
            if "published_after" in filters and doc.published:
                if doc.published < filters["published_after"]:
                    continue
            if "published_before" in filters and doc.published:
                if doc.published > filters["published_before"]:
                    continue

            # ---- score by keyword frequency ----
            body_lower = doc.body.lower()
            score = sum(body_lower.count(kw) for kw in keywords)
            if score == 0:
                continue

            snippet = _best_snippet(doc.body, keywords, context_chars=200)

            results.append(
                {
                    "source_id": doc.source_id,
                    "filename": doc.filename,
                    "source_name": doc.source_name,
                    "relevance_snippet": snippet,
                    "score": score,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_results]


# ======================================================================
# Internal helpers
# ======================================================================

def _parse_frontmatter(raw: str) -> tuple:
    """Extract YAML-like frontmatter and body from a markdown file."""
    meta: Dict[str, str] = {}
    body = raw

    # Frontmatter sits between the first two '---' lines
    parts = re.split(r"^---\s*$", raw, maxsplit=2, flags=re.MULTILINE)
    if len(parts) >= 3:
        fm_block = parts[1]
        body = parts[2].strip()
        for key in ("source", "url", "published", "type"):
            m = re.search(rf"^\s*{key}\s*:\s*(.+)$", fm_block, re.MULTILINE)
            if m:
                meta[key] = m.group(1).strip()
    return meta, body


def _best_snippet(body: str, keywords: List[str], context_chars: int = 200) -> str:
    """Return a ~context_chars snippet around the best keyword match."""
    body_lower = body.lower()
    best_pos = -1
    best_kw = ""
    for kw in keywords:
        pos = body_lower.find(kw)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_kw = kw

    if best_pos == -1:
        return body[:context_chars].strip() + "…"

    start = max(0, best_pos - context_chars // 2)
    end = min(len(body), best_pos + len(best_kw) + context_chars // 2)
    snippet = body[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet = snippet + "…"
    return snippet
