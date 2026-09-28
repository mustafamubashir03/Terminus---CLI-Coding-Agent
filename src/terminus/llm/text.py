"""One place that knows how to read text out of an LLM response.

Three call sites needed this (the planner, the subtask worker, and the /ask
orchestrator) and each had grown its own copy with slightly different rules.
Keeping one implementation means a provider that starts returning a new content
block shape is handled once.
"""


def message_text(message) -> str:
    """Read the text out of a message, or out of a raw content value.

    Accepts either a message object (anything with a ``content`` attribute) or
    the content value itself, so callers do not have to unwrap first.
    Providers return content as either a string or a list of typed blocks; text
    blocks are concatenated and any other block is stringified so reasoning or
    tool payloads still show up rather than vanishing.
    """
    content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                parts.append(text if text is not None else str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)
