from operator import itemgetter
from typing import List

import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, calinski_harabasz_score, silhouette_score, \
    davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from underthesea import word_tokenize

from app.logger import logger
from matcher.clustering.data_prep import TextPreprocess


def agglomerative_clustering(data_frame, n_clusters, **kwargs):
    # Instantiating HAC
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', **kwargs)

    # Fitting
    hac.fit(data_frame)

    # Getting cluster assignments
    return hac.labels_


def k_means_clustering(data_frame, n_clusters, **kwargs):
    # Clustering with different number of clusters
    k_means = KMeans(n_clusters=n_clusters, **kwargs)

    k_means.fit(data_frame)

    return k_means.predict(data_frame)


def ranking(arr: List, is_revers=False):
    temp_arr = [i for i in arr]
    rank = [0 for i in range(len(arr))]
    temp_arr.sort()

    if is_revers:
        temp_arr.reverse()

    idx = 0
    for item in arr:
        rank[idx] = temp_arr.index(item)
        idx += 1

    return rank


def find_best_cluster(cluster_cnt, ch_scores, s_scores, db_scores):
    ch_rank = ranking(ch_scores)
    print("ch_scores")
    print(ch_scores)
    print(ch_rank)
    s_rank = ranking(s_scores)
    print("s_scores")
    print(s_scores)
    print(s_rank)
    db_rank = ranking(db_scores, is_revers=True)
    print("db_scores")
    print(db_scores)
    print(db_rank)
    rank = [0 for i in range(len(cluster_cnt))]
    for i in range(len(cluster_cnt)):
        rank[i] = ch_rank[i] + s_rank[i] + db_rank[i]
    index = rank.index(max(rank))
    return cluster_cnt[index]


def find_best_model_classification_of_new_profile(vector_df):
    # Assigning the split variables
    x = vector_df.drop(["Cluster #"], 1)
    y = vector_df['Cluster #']

    # Train, test, split
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    """
    ### Finding the Best Model
    - Dummy (Baseline Model)
    - KNN
    - SVM
    - NaiveBayes
    - Logistic Regression
    - Adaboost
    """

    # Dummy
    dummy = DummyClassifier(strategy='stratified')

    # KNN
    knn = KNeighborsClassifier()

    # SVM
    svm = SVC(gamma='scale')

    # NaiveBayes
    nb = ComplementNB()

    # Logistic Regression
    lr = LogisticRegression()

    # Adaboost
    adab = AdaBoostClassifier()

    # List of models
    models = [dummy, knn, svm, nb, lr, adab]

    # List of model names
    names = ['Dummy', 'KNN', 'SVM', 'NaiveBayes', 'Logistic Regression', 'Adaboost']

    # Zipping the lists
    classifiers = dict(zip(names, models))

    # Visualization of the different cluster counts
    vector_df['Cluster #'].value_counts().plot(kind='pie', title='Count of Class Distribution')

    """Since we are dealing with an imbalanced dataset _(because each cluster is not guaranteed to have the
    same amount of profiles)_, we will resort to using the __Macro Avg__ and __F1 Score__ for evaluating
    the performances of each model. """

    # Dictionary containing the model names and their scores
    models_f1 = {}

    # Looping through each model's predictions and getting their classification reports
    for name, model in classifiers.items():
        # Fitting the model
        model.fit(x_train, y_train)

        print('\n' + name + ' (Macro Avg - F1 Score):')

        # Classification Report
        report = classification_report(y_test, model.predict(x_test), output_dict=True)
        f1 = report['macro avg']['f1-score']

        # Assigning to the Dictionary
        models_f1[name] = f1

        print(f1)

    # Model with the Best Performance
    best_name_model = max(models_f1, key=models_f1.get)
    best_score = max(models_f1.values())
    print('best model: ', best_name_model, 'Score:', best_score)

    # Fitting the Best Model to our Dataset
    # Fitting the model
    best_model = classifiers[max(models_f1, key=models_f1.get)]
    best_model.fit(x, y)

    return best_model, best_name_model, best_score


def finding_number_of_clusters_refined_data(Vectorizer, data_frame, fn_algorithm_clustering):
    df = data_frame.filter(
        ['introduction', 'age', 'gender', 'hobbies', 'height', 'x', 'y', 'smoking', 'drinking', 'yourKids',
         'religious'])

    # Applying the function to each user bio
    df['introduction'] = df.introduction.apply(tokenize)

    # Looping through the columns and applying the function
    for col in df.columns:
        df[col] = df[col].apply(string_convert)

    df['hobbies'] = df.hobbies.apply(tokenize)

    # Creating the vectorized DF
    vect_df = vectorization(Vectorizer, df, 'hobbies')
    print(vect_df.head())
    vect_df = vectorization(Vectorizer, vect_df, 'introduction')

    scaler = MinMaxScaler()

    vect_df = pd.DataFrame(scaler.fit_transform(vect_df), index=vect_df.index, columns=vect_df.columns)

    # Instantiating PCA
    pca = PCA()

    # Fitting and Transforming the DF
    df_pca = pca.fit_transform(vect_df)

    # Finding the exact number of features that explain at least 99% of the variance in the dataset
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_9 = len(total_explained_variance[total_explained_variance >= .99])
    n_to_reach_9 = vect_df.shape[1] - n_over_9

    print("PCA reduces the # of features from", vect_df.shape[1], 'to', n_to_reach_9)

    # Reducing the dataset to the number of features determined before
    pca = PCA(n_components=n_to_reach_9)

    # Fitting and transforming the dataset to the stated number of features
    df_pca = pca.fit_transform(vect_df)

    # Seeing the variance ratio that still remains after the dataset has been reduced
    print(pca.explained_variance_ratio_.cumsum()[-1])

    # Setting the amount of clusters to test out
    cluster_cnt = [i for i in range(2, 20, 1)]

    # Establishing empty lists to store the scores for the evaluation metrics
    ch_scores = []

    s_scores = []

    db_scores = []

    # The DF for evaluation
    eval_df = df_pca

    # Looping through different iterations for the number of clusters
    for i in cluster_cnt:
        # Clustering with different number of clusters
        cluster_assignments = fn_algorithm_clustering(eval_df, i)

        # Appending the scores to the empty lists
        ch_scores.append(calinski_harabasz_score(eval_df, cluster_assignments))

        s_scores.append(silhouette_score(eval_df, cluster_assignments))

        db_scores.append(davies_bouldin_score(eval_df, cluster_assignments))

    print("\nThe Calinski-Harabasz Score (find max score):")
    cluster_eval(ch_scores, cluster_cnt)

    print("\nThe Silhouette Coefficient Score (find max score):")
    cluster_eval(s_scores, cluster_cnt)

    print("\nThe Davies-Bouldin Score (find minimum score):")
    cluster_eval(db_scores, cluster_cnt)

    k = find_best_cluster(cluster_cnt, ch_scores, s_scores, db_scores)
    print(f"find_best_cluster {k}")
    cluster_assignments = fn_algorithm_clustering(df_pca, k)

    data_frame["Cluster #"] = cluster_assignments
    vect_df['Cluster #'] = cluster_assignments
    return data_frame, vect_df


def scaling(df, input_df):
    """
    Scales the new data with the scaler fitted from the previous data
    """
    scaler = MinMaxScaler()

    scaler.fit(df)

    input_vector = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)

    return input_vector


def scaling_categories(data_frame):
    scaler = MinMaxScaler()

    # Scaling the categories then replacing the old values
    return data_frame[['introduction']].join(
        pd.DataFrame(scaler.fit_transform(data_frame.drop('introduction', axis=1)),
                     columns=data_frame.columns[1:],
                     index=data_frame.index))


def cluster_eval(y, x):
    """
    Plots the scores of a set evaluation metric. Prints out the max and min values of the evaluation scores.
    """

    # Creating a DataFrame for returning the max and min scores for each cluster
    df = pd.DataFrame(columns=['Cluster Score'], index=[i for i in range(2, len(y) + 2)])
    df['Cluster Score'] = y

    print('-' * 20)
    print('Max Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].max()])
    print('\nMin Value:\nCluster #', df[df['Cluster Score'] == df['Cluster Score'].min()])
    print('-' * 20)

    # # Plotting out the scores based on cluster count
    # plt.figure(figsize=(16, 6))
    # plt.style.use('ggplot')
    # plt.plot(x, y)
    # plt.xlabel('# of Clusters')
    # plt.ylabel('Score')
    # plt.show()


def top_similar(df, cluster, vect_df, input_vect, x_user, y_user, min_age_prefer: int, max_age_prefer: int,
                min_height_prefer: int, max_height_prefer: int, gender_prefer: [], distance_prefer, limit=50,
                list_exclude_id=[]):
    """
    Returns the DataFrame containing the top 10 similar profiles to the new data
    """
    # Filtering out the clustered DF
    des_cluster = vect_df[vect_df['Cluster #'] == cluster[0]].drop('Cluster #', 1)

    # Appending the new profile data
    des_cluster = des_cluster.append(input_vect, sort=False)

    # Finding the Top similar or correlated users to the new user
    user_n = input_vect.index[0]

    # Trasnposing the DF so that we are correlating with the index(users) and finding the correlation
    corr = des_cluster.T.corrwith(des_cluster.loc[user_n])

    # Creating a DF with the similar profiles
    top_sim = corr.sort_values(ascending=False)[1:]
    # The Top Profiles
    filter_profile = df.loc[top_sim.index]
    if len(list_exclude_id) > 0:
        filter_profile = filter_profile[~filter_profile['id'].isin(list_exclude_id)]

    a = (x_user, y_user)
    list_x = filter_profile['x'].tolist()
    list_y = filter_profile['y'].tolist()
    list_index = filter_profile.index.values.tolist()

    list_idx_filter_distance = []
    for idx, (x, y) in enumerate(zip(list_x, list_y)):
        b = (x, y)
        dst = distance.euclidean(a, b)
        if dst <= distance_prefer:
            list_idx_filter_distance.append([list_index[idx], dst])
    list_idx_filter_distance = sorted(list_idx_filter_distance, key=itemgetter(1))
    list_index_sorted = [x[0] for x in list_idx_filter_distance]
    filter_profile = filter_profile[filter_profile.index.isin(list_index_sorted)]
    filter_profile = filter_profile.reindex(list_index_sorted)
    filter_profile = filter_profile[filter_profile['age'].between(min_age_prefer, max_age_prefer)]
    filter_profile = filter_profile[filter_profile['height'].between(min_height_prefer, max_height_prefer)]
    filter_profile = filter_profile[filter_profile['gender'].isin(gender_prefer)]

    # Creating a DF with the Top most similar profiles
    if limit > 0:
        top_profile = filter_profile[0:limit]
    else:
        top_profile = filter_profile
    # Converting the floats to ints
    top_profile[top_profile.columns[1:]] = top_profile[top_profile.columns[1:]]

    return top_profile.astype('object')


def string_convert(x):
    """
    First converts the lists in the DF into strings
    """
    if isinstance(x, list):
        return ' '.join(x)
    else:
        return x


def vectorization(Vectorizer, df, column_name):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """
    # Checking if the column name has been removed already
    if column_name not in ['introduction', 'hobbies']:
        return df
    # Instantiating the Vectorizer
    vectorizer = Vectorizer()

    # Fitting the vectorizer to the Bios
    x = vectorizer.fit_transform(df[column_name])

    # Creating a new DF that contains the vectorized words
    df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

    # Concating the words DF with the original DF
    new_df = pd.concat([df, df_wrds], axis=1)

    # Dropping the column because it is no longer needed in place of vectorization
    new_df = new_df.drop(column_name, axis=1)

    return new_df


def vectorization_profile(Vectorizer, df, column_name, input_df):
    """
    Using recursion, iterate through the df until all the categories have been vectorized
    """

    # Checking if the column name has been removed already
    if column_name not in ['introduction', 'hobbies']:
        return df, input_df

    # Encoding columns with respective values
    # Instantiating the Vectorizer
    vectorizer = Vectorizer()

    # Fitting the vectorizer to the columns
    x = vectorizer.fit_transform(df[column_name].values.astype('U'))

    y = vectorizer.transform(input_df[column_name].values.astype('U'))

    # Creating a new DF that contains the vectorized words
    df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())

    y_wrds = pd.DataFrame(y.toarray(), columns=vectorizer.get_feature_names(), index=input_df.index)

    # Concating the words DF with the original DF
    new_df = pd.concat([df, df_wrds], axis=1)

    y_df = pd.concat([input_df, y_wrds], 1)

    # Dropping the column because it is no longer needed in place of vectorization
    new_df = new_df.drop(column_name, axis=1)

    y_df = y_df.drop(column_name, 1)

    return new_df, y_df


def tokenize(text):
    # Lowercasing the words
    text = TextPreprocess.preprocess(text, True)

    text = word_tokenize(text, format="text")
    return text


def get_similar_profile_refined(Vectorizer, data_frame, vect_df, new_profile, model, x_user, y_user,
                                min_age_prefer: int, max_age_prefer: int, min_height_prefer: int,
                                max_height_prefer: int, gender_prefer, distance_prefer, limit=50,
                                list_exclude_id=[]):
    df = data_frame.filter(
        ['introduction', 'age', 'gender', 'hobbies', 'height', 'x', 'y', 'smoking', 'drinking', 'yourKids',
         'religious'])
    new_profile_clone = new_profile.filter(
        ['introduction', 'age', 'gender', 'hobbies', 'height', 'x', 'y', 'smoking', 'drinking', 'yourKids',
         'religious'])
    # Applying the function to each user bio
    df['introduction'] = df.introduction.apply(tokenize)
    new_profile_clone['introduction'] = new_profile_clone.introduction.apply(tokenize)

    # Looping through the columns and applying the string_convert() function (for vectorization purposes)
    for col in df.columns:
        df[col] = df[col].apply(string_convert)
        new_profile_clone[col] = new_profile_clone[col].apply(string_convert)

    df['hobbies'] = df.hobbies.apply(tokenize)
    new_profile_clone['hobbies'] = new_profile_clone.hobbies.apply(tokenize)

    # Creating the vectorized DF
    df_v, input_df = vectorization_profile(Vectorizer, df, 'hobbies', new_profile_clone)
    df_v, input_df = vectorization_profile(Vectorizer, df_v, 'introduction', input_df)

    # Scaling the New Data
    new_df = scaling(df_v, input_df)

    # Predicting/Classifying the new data
    logger.info(new_df)
    cluster = model.predict(new_df)

    logger.info(f"Predicting the New Profile data by determining which Cluster it would belong to: {cluster}")

    # Finding the top related profiles
    top_similar_df = top_similar(data_frame, cluster, vect_df, new_df, x_user, y_user, min_age_prefer,
                                 max_age_prefer,
                                 min_height_prefer, max_height_prefer, gender_prefer, distance_prefer,
                                 limit, list_exclude_id)

    return top_similar_df
