import pickle

import joblib
import pandas as pd
import requests
from celery.exceptions import SoftTimeLimitExceeded

from app.logger import logger
from matcher.clustering.matcher import finding_number_of_clusters_refined_data, k_means_clustering, \
    find_best_model_classification_of_new_profile
from matcher.clustering.vectorizer import vectorizer
from worker.celery_app import celery_app

default_url = 'https://cupizz.cf/export/users?authorization=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI1MjViYTk0NS1iOTA5LTQwZGMtOWFjMi1lNjkyZWYyNDFiNDciLCJpYXQiOjE2MDkxNzAwOTksImV4cCI6MTYwOTI1NjQ5OX0.bptyirh9clkat-jD3On_Snjh_GG_74t4xKK4fP4qw7A'


def train_model_clustering(link_url):
    url = (link_url, default_url)[link_url is None]
    response = requests.get(url)
    if response.ok:
        # json_data = json.load(response.content)
        data_frame = pd.read_json(response.content)
        data_frame.dropna(inplace=True)
        data_frame.reset_index(drop=True, inplace=True)
        print(data_frame.head())
        with open(r"profiles.pkl", 'wb') as wb:
            data_frame.to_csv(r"profiles.csv", encoding='utf-8')
            pickle.dump(data_frame, wb)
        cluster_df, vect_df = finding_number_of_clusters_refined_data(vectorizer, data_frame,
                                                                      fn_algorithm_clustering=k_means_clustering)

        with open(r"cluster.pkl", 'wb') as wb:
            pickle.dump(cluster_df, wb)
            cluster_df.to_csv(r'cluster.csv')

        with open(r"vectorized.pkl", 'wb') as wb:
            pickle.dump(vect_df, wb)
            vect_df.to_csv(r'vectorized.csv')
        best_model, best_name_model, best_score = find_best_model_classification_of_new_profile(vect_df)
        # Saving the Classification Model For future use
        joblib.dump(best_model, "classification_model.joblib")
    else:
        response.raise_for_status()


@celery_app.task(name="clustering")
def train_model(*args, **kwargs) -> str:
    try:
        train_model_clustering(*args, **kwargs)
        return "success"
    except SoftTimeLimitExceeded as ex:
        logger.warning("except SoftTimeLimitExceeded task feeds.googleads.download.update_campaigns")
        return "failed"
