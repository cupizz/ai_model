import pickle
from typing import List

import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import UJSONResponse

from app.response import BaseResponseModel, generate_response
from matcher.clustering.generate_data import create_new_profile
from matcher.clustering.matcher import get_similar_profile_refined
from matcher.clustering.vectorizer import vectorizer

recommend_router = APIRouter(default_response_class=UJSONResponse)


class RecommendInput(BaseModel):
    list_exclude_id: List[str] = []
    id: str
    nickname: str
    introduction: str
    age: int
    gender: int
    height: int
    x: float
    y: float
    smoking: int
    drinking: int
    your_kids: int
    religious: int
    hobbies: list
    min_age_prefer: int
    max_age_prefer: int
    min_height_prefer: int
    max_height_prefer: int
    gender_prefer: List[int]
    distance_prefer: int
    limit: int


def recommend_similar_profile(body: RecommendInput):
    with open("profiles.pkl", 'rb') as dffp:
        df = pickle.load(dffp)

    with open("vectorized.pkl", 'rb') as vect_dffp:
        vect_df = pickle.load(vect_dffp)
    new_profile = create_new_profile(df, body.id, body.nickname, body.introduction, body.age, body.gender,
                                     body.height, body.x, body.y, body.smoking, body.drinking, body.your_kids,
                                     body.religious, body.hobbies)
    model = joblib.load(r"classification_model.joblib")
    list_exclude_id = body.list_exclude_id
    list_exclude_id.append(body.id)
    return get_similar_profile_refined(vectorizer, df, vect_df, new_profile, model, body.x, body.y,
                                       body.min_age_prefer, body.max_age_prefer, body.min_height_prefer,
                                       body.max_height_prefer, body.gender_prefer, body.distance_prefer,
                                       body.limit, list_exclude_id)


@recommend_router.post('/', response_model=BaseResponseModel)
async def train_model_cluster(body: RecommendInput):
    data = recommend_similar_profile(body)
    return generate_response(status_code=200, msg='Success', data=data['id'].tolist())
