"""Configure application logging without recording document or message bodies."""

import logging


def configure_logging(level: str) -> None:
    """Set readable log formatting and silence verbose provider/HTTP libraries."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in ("httpx", "httpcore", "httpx2", "httpcore2", "openai", "pypdf"):
        logging.getLogger(name).setLevel(logging.ERROR)
