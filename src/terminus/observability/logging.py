import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(module_name: str)-> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    return logger
    