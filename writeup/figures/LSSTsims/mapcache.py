"""
mapcache.py: a simple mapped object persistency model, with various backends.
"""

import pickle
from datetime import datetime
import os
import numpy as np

try:
    # Python 2
    import itertools.imap as map
except ImportError:
    pass


class MapCache(object):
    """Base Class for mapped object persistency"""
    def __init__(self, filename):
        raise NotImplementedError()

    def key_exists(self, key):
        raise NotImplementedError()

    def add_row(self, key, val, save=True):
        raise NotImplementedError()

    def get_row(self, key):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def compute(self, func, key, *args, **kwargs):
        if self.key_exists(key):
            return self.get_row(key)
        else:
            return self.add_row(key, func(key, *args, **kwargs))

    def compute_iter(self, func, keys, *args, **kwargs):
        return [self.compute(func, key, *args, **kwargs)
                for key in keys]


class PickleCache(MapCache):
    """Pickle-backed Persistency Helper"""
    def __init__(self, filename):
        self.filename = filename
        self.dct = self.load() if os.path.exists(filename) else {}

    def __iter__(self):
        return iter(self.dct)

    def keys(self):
        return self.dct.keys()

    def values(self):
        return self.dct.values()

    def items(self):
        return self.dct.items()

    def key_exists(self, key):
        return key in self.dct

    def add_row(self, key, val, save=True):
        try:
            self.dct[key] = val
            return val
        finally:
            if save: self.save()

    def get_row(self, key):
        return self.dct[key]

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.dct, f)

    def load(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)


class NumpyCache(PickleCache):
    """Numpy storage-backed Persistency Helper"""
    def __init__(self, filename):
        if not filename.endswith('.npy'):
            filename += '.npy'
        PickleCache.__init__(self, filename)

    def save(self):
        np.save(self.filename, _dict_to_array(self.dct))

    def load(self):
        return _array_to_dict(np.load(self.filename))


#----------------------------------------------------------------------
# Utilities:

def _array_to_dict(arr):
    """Convert mapping array to a dictionary"""
    keys = arr['key']
    vals = arr['val']

    if keys.dtype.names is not None:
        names = keys.dtype.names
        keys = (tuple(key[name] for name in names) for key in keys)
    return dict(zip(keys, vals))


def _dtype_of_key(key):
    """Find dtype associated with the object or tuple"""
    dtypes = ['i8', 'f8', 'c16', '<U32']
    def get_type(val):
        for dtype in dtypes:
            if np.issubdtype(type(val), dtype):
                return dtype
        else:
            raise ValueError("Can't find type for {0}".format(val))

    if isinstance(key, tuple):
        return np.dtype(','.join(map(get_type, key)))
    else:
        return np.dtype(get_type(key))            


def _dict_to_array(dct):
    """Convert dictionary to a mappint array"""
    keys, vals = map(list, zip(*dct.items()))

    keys_dtype = _dtype_of_key(keys[0])
    keys = np.array(keys, dtype=keys_dtype)

    vals = np.asarray(vals)
    vals_dtype = [('key', keys.dtype),
                  ('val', '{0}{1}'.format(vals.shape[1:],
                                          vals.dtype.name))]

    arr = np.zeros(len(keys), dtype=vals_dtype)
    arr['key'] = keys
    arr['val'] = vals
    return arr


#----------------------------------------------------------------------
# Tools for parallel computation
def compute_parallel(cache, func, keys, save_every=4,
                     func_args=None, func_kwargs=None,
                     parallel=True, client=None):
    """Do a parallel computation of a function"""
    keys = [key for key in keys]
    results = dict(cache.items())
    print(50 * '=')
    print("Starting parallel run of {0} results".format(len(keys)))
    print(" - parallel={0}".format(parallel))

    if results:
        print(" - found {0} previous results in {1}"
              "".format(len(results), cache.filename))
    keys_to_compute = [key for key in keys if key not in results]

    # default arguments
    def iter_function(key, func=func, func_args=func_args,
                      func_kwargs=func_kwargs):
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        return func(key, *func_args, **func_kwargs)

    print(" - computing {0} results".format(len(keys_to_compute)))

    # Set up the iterator over results
    if parallel:
        # Use interactive to prevent namespace issues
        from IPython.parallel.util import interactive
        iter_function = interactive(iter_function)
        if client is None:
            from IPython.parallel import Client
            client = Client()
        lbv = client.load_balanced_view()
        results_iter = lbv.map(iter_function, keys_to_compute,
                               block=False, ordered=False)
    else:
        results_iter = map(iter_function, keys_to_compute)

    # Do the iteration, saving the results occasionally
    print(datetime.now())
    for i, (key, result) in enumerate(results_iter):
        print('{0}/{1}: {2}'.format(i + 1, len(keys), result))
        print(' {0}'.format(datetime.now()))
        cache.add_row(key, result, save=((i + 1) % save_every == 0))
    cache.save()

    return np.array([cache.get_row(key) for key in keys])


#----------------------------------------------------------------------
# Unit tests:


def test_inputs_outputs():
    from numpy.testing import assert_equal, assert_allclose

    def check_class(Class, filename, compute, inputs):
        if os.path.exists(filename):
            os.remove(filename)

        # First time: make sure the function is called
        compute.count = 0
        db = Class(filename)
        res1 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 5)

        compute.count = 0
        res2 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 0)
        db.save()

        compute.count = 0
        db2 = Class(filename)
        res3 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 0)

        assert_allclose(res1, res2)
        assert_allclose(res2, res3)

        if os.path.exists(filename):
            os.remove(filename)
    def countevals(f):
        def g(*args, **kwargs):
            g.count += 1
            return f(*args, **kwargs)
        g.count = 0
        return g

    @countevals
    def compute_simple(i):
        return i ** 2

    @countevals
    def compute_simple_arr(i):
        return i ** np.arange(4)

    @countevals
    def compute_multikey(t):
        a, b, c = t
        return [a ** b, float(c)]
    inputs_multi = [(i, (i + 1) / 5, '{0}.{1}'.format(int(i), int(i + 2)))
                    for i in range(5)]

    for compute, inputs in [(compute_simple, range(5)),
                            (compute_simple_arr, range(5)),
                            (compute_multikey, inputs_multi)]:
        for c, f in [(PickleCache, 'tmp.pkl'), (NumpyCache, 'tmp.npy')]:
            yield check_class, c, f, compute, inputs    


def test_utils():
    from numpy.testing import assert_equal
    D = {(1, 2.5, 'abc'): np.arange(0, 3),
         (2, 3.5, 'def'): np.arange(4, 7)}
    arr = _dict_to_array(D)
    D2 = _array_to_dict(arr)
    arr2 = _dict_to_array(D2)
    assert_equal(D, D2)
    assert_equal(arr, arr2)


if __name__ == '__main__':
    import nose; nose.runmodule()
