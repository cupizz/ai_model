import logging

from mongoengine import connect, ConnectionFailure, get_connection

from app.settings import settings

logger = logging.getLogger('mongo.connection')


def get_installed_settings() -> list:
    try:
        installed_apps = settings.INSTALLED_APPS
    except Exception:
        raise ImportError('Please define [env.INSTALLED_APPS] in setting toml file')
    return [getattr(settings, app) for app in installed_apps.APPS]


def get_params_mongo_settings(setting: dict):
    try:
        host = settings.MONGODB_SETTINGS['HOST']
        if settings.MONGODB_SETTINGS['MONGO_MULTI_HOST'] is True:
            host = setting['MONGO_URI_CONNECTION']

        db = setting['MONGO_DB']
        alias = setting['MONGO_ALIAS']
    except Exception as e:
        logger.error(e)
        db = 'default'
        alias = 'default'
        host = ''
    return {'db': db, 'alias': alias, 'host': host}


def get_connect(db='default', alias='default', host='', **kwargs):
    try:
        return connect(
            db, alias=alias, host=host, **kwargs
        )
    except ConnectionFailure:
        return get_connection(alias=alias)


def create_connection(**kwargs):
    conns = []
    for setting in settings.MONGODB_SETTINGS:
        param = get_params_mongo_settings(setting)
        client = get_connect(**param, **kwargs)
        conns.append(client)
        logger.info(f'Mongo Connected: {client} with {param}')
    return conns
