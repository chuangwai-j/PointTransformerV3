# pointcept/utils/registry.py
class Registry:
    """A registry to map strings to classes."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register(self, name=None):
        def _register(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f'{key} is already registered '
                               f'in {self.name}')
            self._module_dict[key] = cls
            return cls

        return _register

    def build(self, cfg, **kwargs):
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

        if 'NAME' not in cfg:
            raise KeyError('cfg must contain the key "NAME"')

        name = cfg['NAME']
        if name not in self._module_dict:
            raise KeyError(f'{name} is not registered in {self.name}')

        cls = self._module_dict[name]

        # Filter out None values
        cfg = {k: v for k, v in cfg.items() if v is not None}

        # Remove 'NAME' key
        cfg.pop('NAME')

        # Instantiate the class
        try:
            instance = cls(**cfg, **kwargs)
        except Exception as e:
            raise type(e)(f'Failed to instantiate {name}: {e}')

        return instance