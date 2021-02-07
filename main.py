import pandas as pd
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from app.app_factory import create_application
from app.logger import logger
from app.mongodb import create_connection
from app.settings import settings
from matcher.clustering.routers.recommend import recommend_router
from matcher.clustering.routers.train import train_router

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

app = create_application(settings)

app.include_router(train_router, tags=['clustering'], prefix='/train')

app.include_router(recommend_router, tags=['recommend'], prefix='/recommend')

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.router.redirect_slashes = False


@app.on_event("startup")
def get_mongo_connection():
    return create_connection()


@app.on_event("shutdown")
def shutdown_event():
    from mongoengine.connection import disconnect_all
    logger.info('Tearing down application')
    disconnect_all()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", reload=True)
