import yaml

class Config(dict):
    def __init__(self, path=None, **kwargs):
        if isinstance(path, str):
            with open(path) as f:
                path = yaml.load(f, Loader=yaml.Loader)
            super().__init__(path)
        else:
            super().__init__(**kwargs)
    
    def __getattr__(self, name):
        value = self[name]
        return value