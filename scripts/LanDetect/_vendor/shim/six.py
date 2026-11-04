"""six shim -- clrnet.utils.registry only needs `string_types`."""

import sys

PY2 = False
PY3 = True

string_types = (str,)
integer_types = (int,)
class_types = (type,)
text_type = str
binary_type = bytes
MAXSIZE = sys.maxsize


def iteritems(d, **kw):
    return iter(d.items(**kw))


def itervalues(d, **kw):
    return iter(d.values(**kw))


def iterkeys(d, **kw):
    return iter(d.keys(**kw))


def u(s):
    return s
