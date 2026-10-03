"""What a vector backend returns: one dict per chunk.

A ``TypedDict`` rather than a class, because retrievers already build a dict and
consumers already read it by key. The field names are the contract:

    text        the chunk's source text
    source      repository-relative file path
    name        the symbol or block name the chunker assigned
    type        the chunk kind ("function", "class", ...)
    start_line  first line, 1-based inclusive
    end_line    last line, 1-based inclusive
    score       relevance, or None when the backend does not expose one

``score`` is always present. Qdrant returns one; Chroma does not. Omitting the
key would make a reader raise KeyError on the backend least likely to be tested,
so it is present and None, meaning "not available" rather than "not relevant".
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class RetrievedChunk(TypedDict):
    """One search result, identical in shape across every vector backend."""

    text: str
    source: str
    name: str
    type: str
    start_line: int
    end_line: int
    score: float | None
    project: NotRequired[str]
    """Which project this chunk came from.

    Optional because Chroma stores each project in its own collection, making the
    project implicit, while Qdrant carries it as a filterable payload field.
    """
