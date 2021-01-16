import _pickle as pickle

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from matcher.clustering.generate_data import gathering_profile_data, generate_new_profile
from matcher.clustering.matcher import k_means_clustering, \
    get_similar_profile_refined, find_best_model_classification_of_new_profile, \
    finding_number_of_clusters_refined_data

Vectorizer = TfidfVectorizer


def test_create_profile_data():
    df = gathering_profile_data()
    with open("../files/pickles/profiles.pkl", 'wb') as wb:
        df.to_csv(r"../files/csv/profiles.csv", encoding='utf-8')
        pickle.dump(df, wb)


def test_finding_number_of_clusters_refined_data():
    # Loading in the cleaned DF
    with open("../files/pickles/profiles.pkl", 'rb') as fp:
        df = pickle.load(fp)
        cluster_df, vect_df = \
            finding_number_of_clusters_refined_data(Vectorizer, df, fn_algorithm_clustering=
            k_means_clustering)

        with open("../files/pickles/cluster.pkl", 'wb') as wb:
            pickle.dump(cluster_df, wb)
            cluster_df.to_csv(r'../files/csv/cluster.csv')

        with open("../files/pickles/vectorized.pkl", 'wb') as wb:
            pickle.dump(vect_df, wb)
            vect_df.to_csv(r'../files/csv/vectorized.csv')


def test_find_best_model_classification_of_new_profile():
    with open("../files/pickles/vectorized.pkl", 'rb') as fp:
        vect_df = pickle.load(fp)
        best_model, best_name_model, best_score = find_best_model_classification_of_new_profile(vect_df)
        # Saving the Classification Model For future use
        dump(best_model, "../files/joblib/refined_model.joblib")


def test_get_similar_profile_refined():
    # Loading the Profiles
    with open("../files/pickles/profiles.pkl", 'rb') as dffp:
        df = pickle.load(dffp)

    with open("../files/pickles/vectorized.pkl", 'rb') as vect_dffp:
        vect_df = pickle.load(vect_dffp)
    new_profile = generate_new_profile(df)
    new_profile.to_csv(r"../files/csv/new_profile.csv")
    print(new_profile.head())
    model = load("../files/joblib/refined_model.joblib")
    get_similar_profile_refined(Vectorizer, df, vect_df, new_profile, model)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    # test_create_profile_data()
    test_finding_number_of_clusters_refined_data()
    # test_find_best_model_classification_of_new_profile()
    # test_get_similar_profile_refined()
    pass
