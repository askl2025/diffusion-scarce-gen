"""
Minimal stand-in for the slice of mmcv that CLRNet touches at inference time.

The real mmcv is not installed and cannot be built against torch 2.0 on
Windows without a compiler toolchain. CLRNet only needs three things from it:

    mmcv.jit               -> decorating accuracy(), training-only
    mmcv.cnn.ConvModule    -> used by FPN / ROIPrior / ROIGather
    mmcv.runner.auto_fp16  -> imported by pafpn.py, never actually applied

Everything here is a faithful re-implementation of the upstream semantics for
the argument combinations CLRNet actually uses, so the vendored model code can
stay byte-for-byte identical to upstream (which keeps the official
checkpoint's key names and shapes valid).
"""

__version__ = '1.2.5-dsh-shim'


def jit(*args, **kwargs):
    """mmcv.jit compiles with numba when available; otherwise it is identity."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def load(file, file_format=None, **kwargs):
    """Only the pickle/json paths are provided; CLRNet uses .py configs."""
    import json
    import pickle

    name = str(file)
    if file_format == 'json' or name.endswith('.json'):
        with open(file, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    with open(file, 'rb') as fh:
        return pickle.load(fh)


def dump(obj, file=None, file_format=None, **kwargs):
    import json
    import pickle

    if file_format == 'json' or (file and str(file).endswith('.json')):
        text = json.dumps(obj, indent=4)
        if file:
            with open(file, 'w', encoding='utf-8') as fh:
                fh.write(text)
            return
        return text
    if file:
        with open(file, 'wb') as fh:
            pickle.dump(obj, fh)
        return
    return pickle.dumps(obj)


from . import cnn  # noqa: E402,F401
from . import runner  # noqa: E402,F401
