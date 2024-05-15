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
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    
    def write(self, path):
        with open(path, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, sort_keys=False)

    def change_keys(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def print_used_keys(self, used_keys):
        unused_keys = set(self) - used_keys
        print("Config used:")
        for i in used_keys:
            name = (i[:18] + '..') if len(i) > 20 else i
            print(name[:20] + " "*(20-len(i)), "\t", self[i])
        print("Config unused:")
        for i in unused_keys:
            name = (i[:18] + '..') if len(i) > 20 else i
            print(name[:20] + " "*(20-len(i)), "\t", self[i])