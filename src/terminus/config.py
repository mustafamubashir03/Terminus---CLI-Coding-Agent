import yaml
from pathlib import Path

def load_config()->dict:
    """Loads configuration from config.yaml file."""
    return yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


CONFIG = load_config()
