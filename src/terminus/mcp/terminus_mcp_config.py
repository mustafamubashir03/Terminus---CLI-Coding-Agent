import os
import re
import json
from pathlib import Path

from terminus.env import load_project_env

_CONFIG_PATH = Path.cwd() / "terminus_mcp_servers.json"


def load_terminus_mcp_config() -> dict:
    """Return MCP server config from the current project."""
    load_project_env(Path.cwd())
    os.environ.setdefault("CWD",str(Path.cwd()))

    if not _CONFIG_PATH.exists():
        return {}

    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    resolved = re.sub(
    r"\$\{(\w+)\}",
    lambda m: json.dumps(os.getenv(m.group(1), ""))[1:-1],
    json.dumps(raw),
    )

    return json.loads(resolved).get("mcp_servers", {})
