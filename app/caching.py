import pickle

import redis

from app.logger import logger
from app.settings import settings

try:
    REDIS = settings.REDIS

except Exception as e:
    raise ImportError('Please make sure import settings for redis')


class Cache(object):
    def __init__(self, host=REDIS['HOST'], port=REDIS['PORT'], **kwargs):
        self.host = host
        self.port = port
        kw_dict = {k.lower(): v for k, v in kwargs.items()}
        self.pool = redis.ConnectionPool(host=host, port=port, **kw_dict)
        self._client = redis.Redis(connection_pool=self.pool)

    def set(self, key, value, **kwargs):
        if isinstance(value, (dict, list)):
            value = pickle.dumps(value)
        self._client.set(key, value, **kwargs)

    def get(self, key, **kwargs):
        value = self._client.get(key)
        try:
            assert value
            value = pickle.loads(value)
        except (pickle.PickleError, AssertionError):
            pass

        return value

    def check_status(self) -> bool:
        status = True
        try:
            self.set("check_status", 1)
            self.get("check_status")
        except (pickle.PickleError, AssertionError):
            status = False

        return status

    def keys(self, pattern='*'):
        return self._client.keys(pattern)

    def delete(self, key):
        return self._client.delete(key)

    def clear(self, pattern='*'):
        for key in self.keys(pattern):
            logger.info(f"Deleting Cache key: {key} by pattern: {pattern}")
            self.delete(key)


cache = Cache()
