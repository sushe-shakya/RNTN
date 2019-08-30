import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG}

log_level = os.getenv("LOG_LEVEL", "info")
logging_level = LOGGING_LEVELS[log_level]
logging.basicConfig(level=logging_level,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
