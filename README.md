Mutiband Lomb-Scargle Periodograms
==================================
This repository contains the source for our multiband periodogram paper.
It makes use of the [gatspy](http://github.com/jakevdp/gatspy/) package,
which has been developed concurrently.
It is a work in progress; to see the current build of the paper,
refer to http://jakevdp.github.io/multiband_LS (powered by [gh-publisher](https://github.com/ewanmellor/gh-publisher)).

Feel free to submit comments or feedback via the Issues tab on this repository.


Reproducing the Paper
---------------------
The LaTeX source of the paper, including all figure pdfs, is in the ``writeup`` directory. The code to reproduce the analysis and figures in the paper is in the ``figures`` directory.

To reproduce the figures, first install the following packages (Python 2 or 3):

- Standard Python scientific stack: ([IPython](http://ipython.org), [numpy](http://numpy.org), [scipy](http://scipy.org), [matplotlib](http://matplotlib.org), [scikit-learn](http://scikit-learn.org), [pandas](http://pandas.pydata.org/))
- [seaborn](http://stanford.edu/~mwaskom/software/seaborn/) for plot styles.
- [astroML](http://astroML.org) for general astronomy machine learning tools.
- [gatspy](http://github.com/astroML/gatspy) for astronomical time-series analysis.
- [supersmoother](http://github.com/jakevdp/supersmoother) for the supersmoother algorithm used by ``gatspy``.

With [conda](http://conda.pydata.org/miniconda.html), a new environment meeting these requirements can be set up as follows:

```
$ conda create -n multibandLS python=3.4 ipython-notebook numpy scipy matplotlib scikit-learn pandas seaborn pip
$ source activate multibandLS
$ pip install astroML gatspy supersmoother
```

Once these packages are installed, navigate to the ``figures`` directory and run any of the ``fig*.py`` scripts. For example, to create figure 1, type
```
$ cd figures
$ python fig01_basic_example.py
```

Several of the figures require the results of long computations. These results are cached as numpy binary files in ``figures/LSSTsims/`` and ``figures/S82sims/``. Code to recompute these results in parallel is in the ``compute_results.py`` script in each of these directories. Note that the full computation for these takes several dozen CPU hours, but is trivially parallelizable with IPython parallel (see scripts for details).