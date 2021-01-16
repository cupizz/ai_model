from fastapi import APIRouter
from starlette.exceptions import HTTPException

from app.caching import cache
from app.logger import logger
from worker.celery_app import celery_app

health_check_router = APIRouter()


def is_celery_working():
    result = celery_app.control.broadcast('ping', reply=True, limit=1)
    return bool(result)  # True if at least one result


@health_check_router.get("/health", tags=["tool"])
def health():
    code = 200
    error_code = 500

    if not cache.check_status():
        raise HTTPException(error_code, "Redis error")
    if is_celery_working():
        logger.info('Celery up!')
    else:
        raise HTTPException(error_code, 'Celery not responding...')
    return {"code": code, "status": "oke"}
