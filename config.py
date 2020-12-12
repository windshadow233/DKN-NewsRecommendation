import json


class Config(object):
    def __init__(self, config_file='config.json'):
        with open(config_file, 'r') as f:
            config_dict = json.loads(f.read())
        for key, value in config_dict.items():
            self.__setattr__(key, value)


config = Config()
