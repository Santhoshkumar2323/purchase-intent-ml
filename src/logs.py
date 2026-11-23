import logging
import os
import warnings

_LEVEL = os.environ.get("PIM_LOG_LEVEL", "WARNING").upper()

warnings.filterwarnings("ignore", message="Unknown solver options", category=UserWarning)
warnings.filterwarnings("ignore", message="Unknown solver options")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, _LEVEL, logging.WARNING))
        logger.propagate = False
    return logger