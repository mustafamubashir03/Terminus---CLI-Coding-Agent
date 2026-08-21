from pathlib import Path
import yaml
import os

DEFAULT_CONFIG = {
    "llm": {
        "provider": "fireworks",
        "model": "gpt-oss-120b"
    },
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "chromadb": {
        "persist_dir": ".chromadb/",
        "collection_name": "terminus"
    },
    "elasticsearch": {
        "url": "http://localhost:9200",
        "index_name": "terminus"
    }
}

def load_config() -> dict:
    """Loads configuration from config.yaml in CWD, package dir, or falls back to defaults."""
    cwd_config = Path.cwd() / "config.yaml"
    pkg_config = Path(__file__).parent / "config.yaml"
    
    if cwd_config.exists():
        try:
            return yaml.safe_load(cwd_config.read_text())
        except Exception:
            pass
            
    if pkg_config.exists():
        try:
            return yaml.safe_load(pkg_config.read_text())
        except Exception:
            pass
            
    return DEFAULT_CONFIG

CONFIG = load_config()
