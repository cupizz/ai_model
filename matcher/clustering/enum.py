import enum


class ClassificationModel(enum.Enum):
    Dummy = 'Dummy'
    KNN = 'KNN'
    SVM = 'SVM'
    NaiveBayes = 'NaiveBayes'
    LogisticRegression = 'LogisticRegression'
    Adaboost = 'Adaboost'


class ClusterModel(enum.Enum):
    KMeans = 'KMeans'
    AgglomerativeClustering = 'AgglomerativeClustering'


class MetricClustering(enum.Enum):
    calinski_harabasz_score = 'ch_scores'
    silhouette_score = 's_scores'
    davies_bouldin_score = 'db_scores'
