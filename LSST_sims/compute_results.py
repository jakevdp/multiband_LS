import os
from datetime import datetime

import numpy as np

from LSSTsims import LSSTsims


def find_res(results, **kwargs):
    mask = np.logical_and.reduce([results[key] == val
                                  for key, val in kwargs.items()], 0)
    return results[mask]['periods'][0]

def value_in_results(results, **kwargs):
    return np.any(np.logical_and.reduce([results[key] == val
                                         for key, val in kwargs.items()], 0))


def compute_and_save_periods(Model, outfile,
                             pointing_indices, ndays,
                             rmags, template_indices,
                             model_args=None, model_kwds=None,
                             Nperiods=5, save_every=5,
                             parallel=True, client=None,
                             num_results=None):
    """Function to compute periods and save the results"""
    # Here is the datatype of the results array:
    dtype = np.dtype([('pointing', 'int32'),
                      ('ndays', 'int32'),
                      ('rmag', 'float32'),
                      ('template', 'int32'),
                      ('periods', '{0:d}float64'.format(Nperiods))])

    keys = list(np.broadcast(pointing_indices, ndays, rmags, template_indices))

    if outfile[-4:] != '.npy':
        outfile += '.npy'

    if os.path.exists(outfile):
        prev_results = np.load(outfile)
        if prev_results.dtype != dtype:
            raise ValueError("dtype of previous results does not match.")
        print("using {0} previous results from {1}"
              "".format(len(prev_results), outfile))
        keys = [key for key in keys
                if not value_in_results(prev_results,
                                        **dict(zip(dtype.names, key)))]
    else:
        prev_results = np.zeros(0, dtype=dtype)

    # For testing purposes... only find a few results
    if num_results is not None:
        keys = keys[:num_results]

    # Define a function which, given a key, computes the desired periods.
    def find_periods(key, Nperiods=Nperiods, Model=Model, LSSTsims=LSSTsims,
                     model_args=model_args, model_kwds=model_kwds):
        lsstsim = LSSTsims()
        t, y, dy, filts = lsstsim.generate_lc(*key, random_state=0)
        model = Model(*(model_args or ()), **(model_kwds or {}))
        model.optimizer.period_range = (0.2, 1.2)
        model.optimizer.verbose = 0
        model.fit(t, y, dy, filts)
        return key + (model.find_best_periods(Nperiods),)

    # Set up the iterator over results
    results = np.zeros(len(keys), dtype=dtype)
    if parallel:
        if client is None:
            from IPython.parallel import Client
            client = Client()
        lbv = client.load_balanced_view()
        results_iter = enumerate(lbv.map(find_periods, keys,
                                         block=False, ordered=False))
    else:
        results_iter = enumerate(map(find_periods, keys))

    # Do the iteration, saving the results occasionally
    print(datetime.now())
    for i, result in results_iter:
        print('{0}/{1}: {2}'.format(i + 1, len(keys), result))
        print(' {0}'.format(datetime.now()))
        results[i] = result

        if (i + 1) % save_every == 0:
            np.save(outfile, np.concatenate([prev_results,
                                             results[:i + 1]]))

    results = np.concatenate([prev_results, results])
    np.save(outfile, results)

    return gather_results(outfile, pointing_indices,
                          ndays, rmags, template_indices)


def gather_results(outfile, pointing_indices, ndays, rmags, template_indices):
    results = np.load(outfile)
    brd = np.broadcast(pointing_indices, ndays, rmags, template_indices)
    dtype = results.dtype
    results = np.array([find_res(results, **dict(zip(dtype.names, key)))
                        for key in brd])
    return results.reshape(brd.shape + results.shape[-1:])
    
                   


if __name__ == '__main__':
    from gatspy.periodic import LombScargleMultiband

    rng = np.random.RandomState(0)
    template_indices = rng.randint(0, 483, 5)
    pointing_indices = np.arange(1, 24)[:, None]
    ndays = np.array([365, 3 * 365, 5 * 365, 10 * 365])[:, None, None]
    rmags = np.array([20, 22, 24.5])[:, None, None, None]

    compute_and_save_periods(LombScargleMultiband, 'results.npy',
                             model_kwds=dict(Nterms_base=1, Nterms_band=0),
                             pointing_indices=pointing_indices,
                             ndays=ndays,
                             rmags=rmags,
                             template_indices=template_indices,
                             parallel=True,
                             save_every=1)
