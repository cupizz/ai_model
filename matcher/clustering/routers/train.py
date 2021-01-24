from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import UJSONResponse

from app.response import BaseResponseModel, generate_response
from matcher.clustering.tasks import train_model_thread

train_router = APIRouter(default_response_class=UJSONResponse)


class TrainInput(BaseModel):
    link_data: str


@train_router.post('/', response_model=BaseResponseModel)
async def train_model_cluster(item: TrainInput):
    train_model_thread(item.link_data)
    return generate_response()
