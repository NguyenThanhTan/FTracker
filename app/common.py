import os


def getEnv(env_name, default_value):
    return os.environ.get(env_name, default_value)
