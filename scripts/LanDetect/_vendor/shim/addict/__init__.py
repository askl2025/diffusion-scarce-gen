"""
Minimal `addict` stand-in: a dict whose keys are also reachable as attributes.

CLRNet uses addict.Dict for its config objects, so nested access like
`cfg.heads.num_priors` and mutation like `cfg['a'] = 1` must both work, and
`.copy()` must stay a Dict (build_from_cfg relies on it).
"""


class Dict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._convert(*args, **kwargs)

    def _convert(self, *args, **kwargs):
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
            else:
                for k, v in dict(arg).items():
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    @staticmethod
    def _wrap(value):
        if isinstance(value, Dict):
            return value
        if isinstance(value, dict):
            return Dict(value)
        if isinstance(value, (list, tuple)):
            wrapped = [Dict._wrap(v) for v in value]
            return type(value)(wrapped) if isinstance(value, tuple) else wrapped
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(item)

    def copy(self):
        return Dict(self)

    def __repr__(self):
        return f'Dict({dict.__repr__(self)})'


class Addict(Dict):
    pass
