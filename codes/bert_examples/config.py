import yaml

class Config(object):
    def __init__(self,config_file):
        with open(config_file,'r',encoding='utf-8') as f:
            cf = yaml.load(f)
        for key,value in cf.items():
            self.__dict__[key] = value