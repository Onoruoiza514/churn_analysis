import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Resolve logs directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Creating custom logger
logger = logging.getLogger("App_Logger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (rotates daily, keeps last 3 logs)
    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=3
    )
    file_handler.setFormatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
