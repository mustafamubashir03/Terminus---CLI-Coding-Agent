from pathlib import Path

import yaml


def load_config()->dict:
    """Loads configuration from config.yaml file."""
    return yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


CONFIG = load_config()
