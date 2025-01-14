# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _DifficultyEstimatorGLinux
else:
    import _DifficultyEstimatorGLinux

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_SwigPyIterator

    def value(self):
        return _DifficultyEstimatorGLinux.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _DifficultyEstimatorGLinux.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _DifficultyEstimatorGLinux.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _DifficultyEstimatorGLinux.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _DifficultyEstimatorGLinux.SwigPyIterator_equal(self, x)

    def copy(self):
        return _DifficultyEstimatorGLinux.SwigPyIterator_copy(self)

    def next(self):
        return _DifficultyEstimatorGLinux.SwigPyIterator_next(self)

    def __next__(self):
        return _DifficultyEstimatorGLinux.SwigPyIterator___next__(self)

    def previous(self):
        return _DifficultyEstimatorGLinux.SwigPyIterator_previous(self)

    def advance(self, n):
        return _DifficultyEstimatorGLinux.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _DifficultyEstimatorGLinux.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _DifficultyEstimatorGLinux.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _DifficultyEstimatorGLinux.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _DifficultyEstimatorGLinux.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _DifficultyEstimatorGLinux.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _DifficultyEstimatorGLinux.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.SwigPyIterator_swigregister(SwigPyIterator)

class StringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _DifficultyEstimatorGLinux.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _DifficultyEstimatorGLinux.StringVector___nonzero__(self)

    def __bool__(self):
        return _DifficultyEstimatorGLinux.StringVector___bool__(self)

    def __len__(self):
        return _DifficultyEstimatorGLinux.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _DifficultyEstimatorGLinux.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _DifficultyEstimatorGLinux.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _DifficultyEstimatorGLinux.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _DifficultyEstimatorGLinux.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _DifficultyEstimatorGLinux.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _DifficultyEstimatorGLinux.StringVector___setitem__(self, *args)

    def pop(self):
        return _DifficultyEstimatorGLinux.StringVector_pop(self)

    def append(self, x):
        return _DifficultyEstimatorGLinux.StringVector_append(self, x)

    def empty(self):
        return _DifficultyEstimatorGLinux.StringVector_empty(self)

    def size(self):
        return _DifficultyEstimatorGLinux.StringVector_size(self)

    def swap(self, v):
        return _DifficultyEstimatorGLinux.StringVector_swap(self, v)

    def begin(self):
        return _DifficultyEstimatorGLinux.StringVector_begin(self)

    def end(self):
        return _DifficultyEstimatorGLinux.StringVector_end(self)

    def rbegin(self):
        return _DifficultyEstimatorGLinux.StringVector_rbegin(self)

    def rend(self):
        return _DifficultyEstimatorGLinux.StringVector_rend(self)

    def clear(self):
        return _DifficultyEstimatorGLinux.StringVector_clear(self)

    def get_allocator(self):
        return _DifficultyEstimatorGLinux.StringVector_get_allocator(self)

    def pop_back(self):
        return _DifficultyEstimatorGLinux.StringVector_pop_back(self)

    def erase(self, *args):
        return _DifficultyEstimatorGLinux.StringVector_erase(self, *args)

    def __init__(self, *args):
        _DifficultyEstimatorGLinux.StringVector_swiginit(self, _DifficultyEstimatorGLinux.new_StringVector(*args))

    def push_back(self, x):
        return _DifficultyEstimatorGLinux.StringVector_push_back(self, x)

    def front(self):
        return _DifficultyEstimatorGLinux.StringVector_front(self)

    def back(self):
        return _DifficultyEstimatorGLinux.StringVector_back(self)

    def assign(self, n, x):
        return _DifficultyEstimatorGLinux.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _DifficultyEstimatorGLinux.StringVector_resize(self, *args)

    def insert(self, *args):
        return _DifficultyEstimatorGLinux.StringVector_insert(self, *args)

    def reserve(self, n):
        return _DifficultyEstimatorGLinux.StringVector_reserve(self, n)

    def capacity(self):
        return _DifficultyEstimatorGLinux.StringVector_capacity(self)
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_StringVector

# Register StringVector in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.StringVector_swigregister(StringVector)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _DifficultyEstimatorGLinux.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _DifficultyEstimatorGLinux.IntVector___nonzero__(self)

    def __bool__(self):
        return _DifficultyEstimatorGLinux.IntVector___bool__(self)

    def __len__(self):
        return _DifficultyEstimatorGLinux.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _DifficultyEstimatorGLinux.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _DifficultyEstimatorGLinux.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _DifficultyEstimatorGLinux.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVector___setitem__(self, *args)

    def pop(self):
        return _DifficultyEstimatorGLinux.IntVector_pop(self)

    def append(self, x):
        return _DifficultyEstimatorGLinux.IntVector_append(self, x)

    def empty(self):
        return _DifficultyEstimatorGLinux.IntVector_empty(self)

    def size(self):
        return _DifficultyEstimatorGLinux.IntVector_size(self)

    def swap(self, v):
        return _DifficultyEstimatorGLinux.IntVector_swap(self, v)

    def begin(self):
        return _DifficultyEstimatorGLinux.IntVector_begin(self)

    def end(self):
        return _DifficultyEstimatorGLinux.IntVector_end(self)

    def rbegin(self):
        return _DifficultyEstimatorGLinux.IntVector_rbegin(self)

    def rend(self):
        return _DifficultyEstimatorGLinux.IntVector_rend(self)

    def clear(self):
        return _DifficultyEstimatorGLinux.IntVector_clear(self)

    def get_allocator(self):
        return _DifficultyEstimatorGLinux.IntVector_get_allocator(self)

    def pop_back(self):
        return _DifficultyEstimatorGLinux.IntVector_pop_back(self)

    def erase(self, *args):
        return _DifficultyEstimatorGLinux.IntVector_erase(self, *args)

    def __init__(self, *args):
        _DifficultyEstimatorGLinux.IntVector_swiginit(self, _DifficultyEstimatorGLinux.new_IntVector(*args))

    def push_back(self, x):
        return _DifficultyEstimatorGLinux.IntVector_push_back(self, x)

    def front(self):
        return _DifficultyEstimatorGLinux.IntVector_front(self)

    def back(self):
        return _DifficultyEstimatorGLinux.IntVector_back(self)

    def assign(self, n, x):
        return _DifficultyEstimatorGLinux.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _DifficultyEstimatorGLinux.IntVector_resize(self, *args)

    def insert(self, *args):
        return _DifficultyEstimatorGLinux.IntVector_insert(self, *args)

    def reserve(self, n):
        return _DifficultyEstimatorGLinux.IntVector_reserve(self, n)

    def capacity(self):
        return _DifficultyEstimatorGLinux.IntVector_capacity(self)
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_IntVector

# Register IntVector in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.IntVector_swigregister(IntVector)

class IntVectorVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _DifficultyEstimatorGLinux.IntVectorVector___nonzero__(self)

    def __bool__(self):
        return _DifficultyEstimatorGLinux.IntVectorVector___bool__(self)

    def __len__(self):
        return _DifficultyEstimatorGLinux.IntVectorVector___len__(self)

    def __getslice__(self, i, j):
        return _DifficultyEstimatorGLinux.IntVectorVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _DifficultyEstimatorGLinux.IntVectorVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector___setitem__(self, *args)

    def pop(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_pop(self)

    def append(self, x):
        return _DifficultyEstimatorGLinux.IntVectorVector_append(self, x)

    def empty(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_empty(self)

    def size(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_size(self)

    def swap(self, v):
        return _DifficultyEstimatorGLinux.IntVectorVector_swap(self, v)

    def begin(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_begin(self)

    def end(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_end(self)

    def rbegin(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_rbegin(self)

    def rend(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_rend(self)

    def clear(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_clear(self)

    def get_allocator(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_get_allocator(self)

    def pop_back(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_pop_back(self)

    def erase(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector_erase(self, *args)

    def __init__(self, *args):
        _DifficultyEstimatorGLinux.IntVectorVector_swiginit(self, _DifficultyEstimatorGLinux.new_IntVectorVector(*args))

    def push_back(self, x):
        return _DifficultyEstimatorGLinux.IntVectorVector_push_back(self, x)

    def front(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_front(self)

    def back(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_back(self)

    def assign(self, n, x):
        return _DifficultyEstimatorGLinux.IntVectorVector_assign(self, n, x)

    def resize(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector_resize(self, *args)

    def insert(self, *args):
        return _DifficultyEstimatorGLinux.IntVectorVector_insert(self, *args)

    def reserve(self, n):
        return _DifficultyEstimatorGLinux.IntVectorVector_reserve(self, n)

    def capacity(self):
        return _DifficultyEstimatorGLinux.IntVectorVector_capacity(self)
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_IntVectorVector

# Register IntVectorVector in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.IntVectorVector_swigregister(IntVectorVector)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _DifficultyEstimatorGLinux.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _DifficultyEstimatorGLinux.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _DifficultyEstimatorGLinux.DoubleVector___bool__(self)

    def __len__(self):
        return _DifficultyEstimatorGLinux.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _DifficultyEstimatorGLinux.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _DifficultyEstimatorGLinux.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _DifficultyEstimatorGLinux.DoubleVector_pop(self)

    def append(self, x):
        return _DifficultyEstimatorGLinux.DoubleVector_append(self, x)

    def empty(self):
        return _DifficultyEstimatorGLinux.DoubleVector_empty(self)

    def size(self):
        return _DifficultyEstimatorGLinux.DoubleVector_size(self)

    def swap(self, v):
        return _DifficultyEstimatorGLinux.DoubleVector_swap(self, v)

    def begin(self):
        return _DifficultyEstimatorGLinux.DoubleVector_begin(self)

    def end(self):
        return _DifficultyEstimatorGLinux.DoubleVector_end(self)

    def rbegin(self):
        return _DifficultyEstimatorGLinux.DoubleVector_rbegin(self)

    def rend(self):
        return _DifficultyEstimatorGLinux.DoubleVector_rend(self)

    def clear(self):
        return _DifficultyEstimatorGLinux.DoubleVector_clear(self)

    def get_allocator(self):
        return _DifficultyEstimatorGLinux.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _DifficultyEstimatorGLinux.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _DifficultyEstimatorGLinux.DoubleVector_swiginit(self, _DifficultyEstimatorGLinux.new_DoubleVector(*args))

    def push_back(self, x):
        return _DifficultyEstimatorGLinux.DoubleVector_push_back(self, x)

    def front(self):
        return _DifficultyEstimatorGLinux.DoubleVector_front(self)

    def back(self):
        return _DifficultyEstimatorGLinux.DoubleVector_back(self)

    def assign(self, n, x):
        return _DifficultyEstimatorGLinux.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _DifficultyEstimatorGLinux.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _DifficultyEstimatorGLinux.DoubleVector_reserve(self, n)

    def capacity(self):
        return _DifficultyEstimatorGLinux.DoubleVector_capacity(self)
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_DoubleVector

# Register DoubleVector in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.DoubleVector_swigregister(DoubleVector)

class Edge(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    s = property(_DifficultyEstimatorGLinux.Edge_s_get, _DifficultyEstimatorGLinux.Edge_s_set)
    _from = property(_DifficultyEstimatorGLinux.Edge__from_get, _DifficultyEstimatorGLinux.Edge__from_set)
    nxt = property(_DifficultyEstimatorGLinux.Edge_nxt_get, _DifficultyEstimatorGLinux.Edge_nxt_set)

    def __init__(self, s, _from, nxt):
        _DifficultyEstimatorGLinux.Edge_swiginit(self, _DifficultyEstimatorGLinux.new_Edge(s, _from, nxt))
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_Edge

# Register Edge in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.Edge_swigregister(Edge)
cvar = _DifficultyEstimatorGLinux.cvar
maxnode = cvar.maxnode

class DifficultyEstimator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    item_freq = property(_DifficultyEstimatorGLinux.DifficultyEstimator_item_freq_get, _DifficultyEstimatorGLinux.DifficultyEstimator_item_freq_set)
    G = property(_DifficultyEstimatorGLinux.DifficultyEstimator_G_get, _DifficultyEstimatorGLinux.DifficultyEstimator_G_set)
    itemset_now = property(_DifficultyEstimatorGLinux.DifficultyEstimator_itemset_now_get, _DifficultyEstimatorGLinux.DifficultyEstimator_itemset_now_set)
    active_edge = property(_DifficultyEstimatorGLinux.DifficultyEstimator_active_edge_get, _DifficultyEstimatorGLinux.DifficultyEstimator_active_edge_set)

    def clear(self):
        return _DifficultyEstimatorGLinux.DifficultyEstimator_clear(self)

    def __init__(self, item_sets, item_freq, n_samples):
        _DifficultyEstimatorGLinux.DifficultyEstimator_swiginit(self, _DifficultyEstimatorGLinux.new_DifficultyEstimator(item_sets, item_freq, n_samples))

    def predict_easy(self, s):
        return _DifficultyEstimatorGLinux.DifficultyEstimator_predict_easy(self, s)

    def predict_and_add(self, s):
        return _DifficultyEstimatorGLinux.DifficultyEstimator_predict_and_add(self, s)
    __swig_destroy__ = _DifficultyEstimatorGLinux.delete_DifficultyEstimator

# Register DifficultyEstimator in _DifficultyEstimatorGLinux:
_DifficultyEstimatorGLinux.DifficultyEstimator_swigregister(DifficultyEstimator)



