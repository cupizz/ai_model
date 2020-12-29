import logging

from app.settings import settings

logger = logging.getLogger(settings.APP_NAME)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.setLevel(settings.LOG_LEVEL)
logger.addHandler(c_handler)
