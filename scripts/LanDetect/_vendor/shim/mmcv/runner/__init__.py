"""mmcv.runner shim -- pafpn.py imports auto_fp16 but never applies it."""

from functools import wraps


def auto_fp16(apply_to=None, out_fp32=False):
    """No-op decorator; CLRNet's FPN path is used, not PAFPN."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def load_checkpoint(*args, **kwargs):
    raise NotImplementedError(
        "mmcv shim: use clrnet.utils.net_utils.load_checkpoint instead")
