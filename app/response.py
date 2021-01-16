from typing import Union, Any, List

from fastapi.responses import UJSONResponse
from pydantic import BaseModel


class BaseResponseModel(BaseModel):
    message: str = "Success"
    data: Union[List[Any], dict] = []


def generate_response(status_code=200, msg='Success', data=()):
    reps = {
        'message': msg,
        'data': data
    }
    return UJSONResponse(status_code=status_code, content=reps)
