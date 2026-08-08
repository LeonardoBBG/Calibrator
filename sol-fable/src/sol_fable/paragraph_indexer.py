"""Public paragraph-indexing interface kept separate from file parsing concerns."""

from .ingest import TextBlock, index_blocks

__all__ = ["TextBlock", "index_blocks"]

