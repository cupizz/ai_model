import datetime

from mongoengine import Document, DateTimeField, EnumField, IntField, FloatField, ListField

from matcher.clustering.enum import ClassificationModel, ClusterModel, MetricClustering


class Report(Document):
    best_number_cluster: IntField()
    cluster_name: EnumField(ClusterModel)
    classification_model_name: EnumField(ClassificationModel)
    pca: FloatField()
    start_time = DateTimeField(default=datetime.utcnow)
    end_time = DateTimeField(default=datetime.utcnow)


class MetricsReport(Document):
    name: EnumField(MetricClustering)
    best_value_cluster: IntField()
    best_value_score: FloatField()
    value: ListField(field=FloatField())
    rank: ListField(field=IntField())
    created_at = DateTimeField(default=datetime.utcnow)
