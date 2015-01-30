import pickle
import os
import numpy as np

from numpy.testing import assert_equal, assert_allclose


class CacheDB(object):
    def __init__(self, filename):
        raise NotImplementedError()

    def key_exists(self, key):
        raise NotImplementedError()

    def add_row(self, key, val, save=False):
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


class PickleCache(CacheDB):
    """Pickle-backed Persistency Helper"""
    def __init__(self, filename):
        self.filename = filename
        self.db = self.load() if os.path.exists(filename) else {}

    def key_exists(self, key):
        return key in self.db

    def add_row(self, key, val, save=False):
        try:
            self.db[key] = val
            return val
        finally:
            if save:
                self.save()

    def get_row(self, key):
        return self.db[key]

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.db, f)

    def load(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)


class NumpyCache(PickleCache):
    """Numpy Binary-backed Persistency Helper"""
    def __init__(self, filename):
        if not filename.endswith('.npy'):
            filename += '.npy'
        PickleCache.__init__(self, filename)

    def save(self):
        np.save(self.filename, self.dict_to_array(self.db))

    def load(self):
        return self.array_to_dict(np.load(self.filename))

    @staticmethod
    def array_to_dict(arr):
        keys = arr['key']
        vals = arr['val']

        if keys.dtype.names is not None:
            names = keys.dtype.names
            keys = (tuple(key[name] for name in names) for key in keys)
        return dict(zip(keys, vals))

    @staticmethod
    def dtype_of_key(key):
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

    @staticmethod
    def dict_to_array(dct):
        keys, vals = map(list, zip(*dct.items()))

        keys_dtype = NumpyCache.dtype_of_key(keys[0])
        keys = np.array(keys, dtype=keys_dtype)

        vals = np.asarray(vals)
        vals_dtype = [('key', keys.dtype),
                      ('val', '{0}{1}'.format(vals.shape[1:],
                                              vals.dtype.name))]

        arr = np.zeros(len(keys), dtype=vals_dtype)
        arr['key'] = keys
        arr['val'] = vals
        return arr
            

def test_pickle():
    def compute(i):
        compute.count += 1
        return i ** 2
    compute.count = 0

    def check_class(Class, filename):
        if os.path.exists(filename):
            os.remove(filename)

        compute.count = 0
        db = Class(filename)
        res1 = db.compute_iter(compute, range(5))
        assert_equal(compute.count, 5)

        compute.count = 0
        res2 = db.compute_iter(compute, range(5))
        assert_equal(compute.count, 0)
        db.save()

        db2 = Class(filename)
        res3 = db.compute_iter(compute, range(5))
        assert_equal(compute.count, 0)

        assert_allclose(res1, res2)
        assert_allclose(res2, res3)

        if os.path.exists(filename):
            os.remove(filename)

    for c, f in [(PickleCache, 'tmp.pkl'), (NumpyCache, 'tmp.npy')]:
        yield check_class, c, f


def test_multi_key():
    def compute(t):
        compute.count += 1
        a, b, c = t
        return [a ** b, float(c)]
    compute.count = 0

    def check_class(Class, filename):
        if os.path.exists(filename):
            os.remove(filename)

        inputs = [(i, (i + 1) / 5, '{0}.{1}'.format(int(i), int(i + 2)))
                  for i in range(5)]

        compute.count = 0
        db = Class(filename)
        res1 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 5)

        compute.count = 0
        res2 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 0)
        db.save()

        db2 = Class(filename)
        res3 = db.compute_iter(compute, inputs)
        assert_equal(compute.count, 0)

        assert_allclose(res1, res2)
        assert_allclose(res2, res3)

        if os.path.exists(filename):
            os.remove(filename)

    for c, f in [(PickleCache, 'tmp.pkl'), (NumpyCache, 'tmp.npy')]:
        yield check_class, c, f

    


def test_dict_to_array():
    D = {(1, 2.5, 'abc'): np.arange(0, 3),
         (2, 3.5, 'def'): np.arange(4, 7)}
    arr = NumpyCache.dict_to_array(D)
    D2 = NumpyCache.array_to_dict(arr)
    arr2 = NumpyCache.dict_to_array(D2)
    assert_equal(D, D2)
    assert_equal(arr, arr2)


if __name__ == '__main__':
    import nose; nose.runmodule()
