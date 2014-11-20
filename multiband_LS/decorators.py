# hash which appropriately hashes numpy arrays
from sklearn.externals.joblib import hash
import functools


def cache_results(arg, filename=None, overwrite=False):
    """Generator for a decorator which allows cacheing results"""
    if type(arg) == str:
        if filename is None:
            raise ValueError("unrecognized input!")
        return functools.partial(cache_results, filename=arg, overwrite=False)
    elif filename is None:
        raise ValueError("must include filename")

    func = arg

    if not callable(func):
        raise ValueError("cache_results must be applied to a function")

    def new_func(*args, **kwargs):
        arg_hash = hash([args, kwargs])

        if os.path.exists(filename):
            cached = np.load(filename)
            if cached['arg_hash'].flat[0] == arg_hash:
                nretvals = cached['nreturns']
                if cached['nretvals'] > 1:
                    return tuple(cached['retvals'])
                else:
                    return cached['retvals']
            elif not overwrite:
                raise ValueError("Cache file exists and input does not match. "
                                 "Set overwrite=True to force overwrite")

            # otherwise, do the computation
            retvals = func(*args, **kwargs)
                
            if type(retvals) is tuple:
                nretvals = len(retvals)
            else:
                nretvals = 1
