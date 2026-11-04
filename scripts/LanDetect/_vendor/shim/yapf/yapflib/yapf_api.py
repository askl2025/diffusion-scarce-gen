"""
yapf.yapflib.yapf_api shim.

Only `FormatCode` is imported by clrnet.utils.config, and only inside
`Config.pretty_text` / `Config.dump`, which the inference path never calls.
It therefore returns the source unchanged.
"""


def FormatCode(source, *args, **kwargs):
    """Return (formatted_source, changed). No formatting performed."""
    return source, False


def FormatFile(*args, **kwargs):
    raise NotImplementedError("yapf shim: FormatFile is not implemented")
