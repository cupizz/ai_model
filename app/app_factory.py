import traceback

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from mongoengine import DoesNotExist, NotUniqueError
from pydantic.error_wrappers import ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import UJSONResponse

from app.health.router import health_check_router
from app.logger import logger


async def validation_exception_response(request: Request, exc: ValidationError):
    # logger.debug(traceback.format_exc())
    print(traceback.format_exc())
    error = exc.errors()[0]
    try:
        field = error['loc'][1]
    except IndexError:
        field = ''
    err_type = error['type']
    if err_type == 'type_error.enum':
        msg = error['msg'].replace('value', field, 1)
    else:
        msg = f"{error['msg']}: {field}"
    return UJSONResponse(
        status_code=400,
        content={
            'status': 'error',
            'message': msg
        },
    )


def create_application(settings) -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get('CORS_ORIGIN', []),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_check_router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if exc.status_code == 500:
            logger.exception(exc)
        else:
            logger.info(exc)
        return UJSONResponse({'message': exc.detail}, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_exception_handler(request: Request, exc: RequestValidationError):
        return await validation_exception_response(request, exc)

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return await validation_exception_response(request, exc)

    @app.exception_handler(DoesNotExist)
    async def mongo_object_not_found_handler(request: Request, exc: DoesNotExist):
        return UJSONResponse({'message': str(exc)}, 404)

    @app.exception_handler(NotUniqueError)
    async def mongo_object_duplicate_handler(request: Request, exc: NotUniqueError):
        return UJSONResponse({'message': str(exc)}, 409)

    @app.get("/", tags=["tool"])
    async def home_res():
        return "success"

    return app
