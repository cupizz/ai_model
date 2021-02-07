import pickle
from threading import Lock

import joblib


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ModelSingleton(metaclass=SingletonMeta):
    __df: None
    __vect_df: None
    __model: None
    __cluster_df: None
    """
    We'll use this property to prove that our Singleton really works.
    """

    def __init__(self) -> None:
        self.__df = None
        self.__vect_df = None
        self.__model = None
        self.__cluster_df = None
        pass

    def get_df(self):
        if self.__df is None:
            with open(r"files/profiles.pkl", 'rb') as dffp:
                self.__df = pickle.load(dffp)
        return self.__df

    def get_vect_df(self):
        if self.__vect_df is None:
            with open("files/vectorized.pkl", 'rb') as vect_dffp:
                self.__vect_df = pickle.load(vect_dffp)
        return self.__vect_df

    def get_cluster_df(self):
        if self.__cluster_df is None:
            with open("files/cluster.pkl", 'rb') as cluster_dffp:
                self.__cluster_df = pickle.load(cluster_dffp)
        return self.__cluster_df

    def get_model(self):
        if self.__model is None:
            self.__model = joblib.load(r"files/classification_model.joblib")
        return self.__model

    def set_df(self, df):
        self.__df = df
        with open(r"files/profiles.pkl", 'wb') as wb:
            df.to_csv(r"files/profiles.csv", encoding='utf-8')
            pickle.dump(df, wb)

    def set_vect_df(self, vect_df):
        self.__vect_df = vect_df
        with open(r"files/vectorized.pkl", 'wb') as wb:
            pickle.dump(vect_df, wb)
            vect_df.to_csv(r'files/vectorized.csv')

    def set_cluster_df(self, cluster_df):
        self.__cluster_df = cluster_df
        with open(r"files/cluster.pkl", 'wb') as wb:
            pickle.dump(cluster_df, wb)
            cluster_df.to_csv(r'files/cluster.csv')

    def set_model(self, model):
        self.__model = model
        joblib.dump(model, "files/classification_model.joblib")


model_singleton = ModelSingleton()
