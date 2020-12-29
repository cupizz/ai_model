import os

from dotenv import load_dotenv
from dynaconf import Dynaconf

env_path = os.getenv('DOTENV_PATH', default='/app/.env')
load_dotenv(dotenv_path='/app/.env')
settings = Dynaconf(
    environments=True,
    envvar_prefix='CONF',
    dotenv_path='/app/.env',
)
