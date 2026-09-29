"""What a vector backend returns to the rest of Terminus.

A ``TypedDict`` rather than a class: the retrievers already build a dict per
chunk, consumers already read it by key, and adding a type would mean converting
at every boundary for no gain. What it buys is that the *shape* is written down
once, and a test can assert every backend honours it.

The shape
---------
``text``          the chunk's source text
``source``        repository-relative file path
``name``          the symbol or block name the chunker assigned
``type``          the chunk kind ("function", "class", ...)
``start_line``    first line, 1-based inclusive
``end_line``      last line, 1-based inclusive
``score``         relevance, or ``None`` when the backend does not expose one

On ``score``
------------
Qdrant's ``similarity_search_with_score`` returns one; Chroma's query path as
written does not. Rather than fabricate a number - or silently omit the key, which
is what Chroma did, so that any consumer reading ``score`` would raise
``KeyError`` on exactly the backend least likely to be tested - the field is
present and ``None``.

``None`` means "not available", not "not relevant". A caller that ranks or
thresholds on score must handle it, and :func:`require_score` exists for the ones
that cannot.
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
    """Present when the backend's stored metadata carried it.

    Optional because Chroma's per-project storage makes the project implicit -
    the collection is the isolation boundary - while Qdrant carries it as a
    payload field it can filter on.
    """


def require_score(chunk: RetrievedChunk) -> float:
    """The score of *chunk*, or a clear error if the backend did not supply one."""
    score = chunk.get("score")
    if score is None:
        raise ValueError(
            f"this vector backend did not return a relevance score for "
            f"{chunk.get('source')}:{chunk.get('start_line')}; it cannot be used "
            f"for ranking or thresholding"
        )
    return score
