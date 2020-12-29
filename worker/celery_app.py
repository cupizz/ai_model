import logging

from celery import Celery, Task
from celery.signals import after_setup_logger

from app.mongodb import create_connection
from app.settings import settings


class AppTask(Task):

    def __call__(self, *args, **kwargs):
        self.connections = create_connection()
        self.push_request(args=args, kwargs=kwargs)
        try:
            return self.run(*args, **kwargs)
        finally:
            self.pop_request()

    def after_return(self, *args, **kwargs):
        for c in self.connections:
            c.close()
        return super().after_return(*args, **kwargs)


def get_celery_settings():
    return {k.lower(): v for k, v in settings.CELERY.items()}


celery_app = Celery(
    settings.APP_NAME,
    task_cls=AppTask
)

celery_app.conf.update(**get_celery_settings())

# find packages which have `tasks.py`
celery_app.autodiscover_tasks(['matcher.clustering'])

logger = logging.getLogger(__name__)


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add filehandler
    fh = logging.FileHandler('logs.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
