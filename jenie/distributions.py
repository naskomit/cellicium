import sys
import traceback
import math

import numpy as np
import distfit
import multiprocessing as mp
import matplotlib.pyplot as plt
from .multiproc import exception2either


def bootstrap(data, bootnum = 100, samples = None, bootfunc = None, funcoutdims = None):
    """Performs bootstrap resampling on numpy arrays.
    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.
    Parameters
    ----------
    data : ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned
    Returns
    -------
    boot : ndarray
        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.
    Examples
    --------
    Obtain a twice resampled array:
    >>> from astropy.stats import bootstrap
    >>> import numpy as np
    >>> from astropy.utils import NumpyRNGContext
    >>> bootarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[6., 9., 0., 6., 1., 1., 2., 8., 7., 0.],
           [3., 5., 6., 3., 5., 3., 5., 8., 8., 0.]])
    >>> bootresult.shape
    (2, 10)
    Obtain a statistic on the array
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2, bootfunc=np.mean)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([4. , 4.6])
    Obtain a statistic with two outputs on the array
    >>> test_statistic = lambda x: (np.sum(x), np.mean(x))
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=test_statistic)
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[40. ,  4. ],
           [46. ,  4.6],
           [35. ,  3.5]])
    >>> bootresult.shape
    (3, 2)
    Obtain a statistic with two outputs on the array, keeping only the first
    output
    >>> bootfunc = lambda x:test_statistic(x)[0]
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=bootfunc)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([40., 46., 35.])
    >>> bootresult.shape
    (3,)
    """
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        if funcoutdims is None:
            try:
                resultdims = (bootnum, len(bootfunc(data)))
            except TypeError:
                resultdims = (bootnum,)
        else:
            resultdims = (bootnum, funcoutdims)

    # create empty boot array
    boot = np.empty(resultdims)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot





class DistFitter():
    def __init__(self, param):
        if 'distname' in param:
            self.distname = param['distname']
        else:
            self.distname = \
                ['beta', 'burr', 'expon', #'birnbaumsaunders',
                 'gamma', 'genextreme', #'extreme value'
                 'pareto', 'invgauss', 'logistic',
                 'lognorm', 'nakagami', 'norm', #'loglogistic'
                 'rayleigh', 'weibull_min', 'weibull_max'] #'rician' 'tlocationscale'

        if 'raw_data' in param:
            self.raw_data = param['raw_data']
        else:
            raise ValueError('No key `raw_data` found in input parameters!')

        self.bootnum = param.get('bootnum', 100)

    def select_optimal(self):
        self.distfit = distfit.distfit(distr = self.distname)
        self.distfit.fit_transform(self.raw_data, verbose=2)
        self.optimal_model = self.distfit.model



    def bootstrap(self):
        distrib_name = self.optimal_model['name']
        #num_model_params = len(self.optimal_model['params'])
        bst_num_samples = max(len(self.raw_data) // 10, 100)

        bstr_objs = []
        for i in range(self.bootnum):
            boot_sample_idx = np.random.randint(low = 0, high = self.raw_data.shape[0], size = bst_num_samples)
            bst_disfit = distfit.distfit(distr = distrib_name)
            try:
                bst_disfit.fit_transform(self.raw_data[boot_sample_idx], verbose=2)
                bstr_objs.append(bst_disfit)
            except Exception:
                pass

#         def bst_fit(data):
#             try:
#                 bst_disfit.fit_transform(data, verbose=2)
#                 return bst_disfit #.model['params']
#             except Exception:
#                 return np.NaN
#         bst_params = bootstrap(
#             self.raw_data, bootnum = self.bootnum, samples = bst_num_samples,
#             bootfunc = bst_fit, funcoutdims = num_model_params
#         )
#         # Remove failed
#         bst_params = bst_params[~np.any(np.isnan(bst_params), axis = 1), :]

        return bstr_objs

    # bootstrapParams, type, data, precision
    def minimize_rectangle(self, bstr_objs, precision):
        num_bst = len(bstr_objs)
        raw_max = np.max(self.raw_data)
        raw_min = np.min(self.raw_data)
        N_div = 10000
        x_scan = np.linspace(0, raw_max, N_div)
        x_off = np.zeros(num_bst, dtype = 'float')
        x_on = np.zeros(num_bst, dtype = 'float')
        #fig, ax = plt.subplots(1, 3)
        for i in range(num_bst):
            y = bstr_objs[i].model['model'].cdf(x_scan)
            off_areas = x_scan * (1 - y)
            on_areas = (raw_max - x_scan) * y
            ind_max_off = np.argmax(off_areas)
            ind_max_on = np.argmax(on_areas)
            x_off[i] = x_scan[ind_max_off]
            x_on[i] = x_scan[ind_max_on]
            # take the minimum observed gene expression value if no extreme
            # point is detected for the lower threshold
            if (x_off[i] > x_on[i]):
                x_off[i] = raw_min

        self.x_off = x_off
        self.x_on = x_on
        return x_off, x_on

    def plot1(self, bstr_objs):
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(x_scan, y_global)
        display(fig)
        print("Plotted?")

#         plt.show
#         num_bst = len(bstr_objs)

#         for i in range(num_bst):
#             y = bstr_objs[i].model['model'].cdf(x_scan)
        #x_scan = np.logspace(np.log10(raw_min), np.log10(raw_max), 100)


#         plt.hist(x_off)
#         #ax[2].hist(x_on)
#         plt.show()

#         max_vals = 10000
#        num_intervals = math.ceil(raw_max / (precision * max_vals))
#         best_low = np.zeros((num_bst, 2))
#         best_high = np.zeros((num_bst, 2))
#         print(f"raw_max: {raw_max}; num_intervals: {num_intervals}" )
#        x_last = 0
#         # Scan each domain interval
#         for j in range(num_intervals):
#             x_cur = x_last + (min((j + 1) * max_vals,  raw_max / precision) - j * max_vals) * precision

#             print(f"x_last: {x_last}, x_cur {x_cur}, min {min((j + 1) * max_vals,  raw_max / precision)}" )
#             x = np.arange(x_last, x_cur, precision)
#             y = np.zeros((num_bst, x.shape[0]))

#             # Compute the values predicted by each sample CDF
#             for i in range(num_bst):
#                 y[i, :] = bstr_objs[i].model['model'].cdf(x)

#             off_areas = x * (1 - y)
#             on_areas = (raw_max - x) * y
#             ind_max_off = np.argmax(off_areas, axis = 1)
#             ind_max_on = np.argmax(on_areas, axis = 1)

#             #print(ind_max_off.shape, off_areas.T.shape)
#             # TODO There should be a way to optimize this
#             area_max_off = np.array([off_areas[i, ind_max_off[i]] for i in range(num_bst)])
#             area_max_on = np.array([on_areas[i, ind_max_on[i]] for i in range(num_bst)])


#             print(area_max_off)

#             x_last = x_cur

#         plt.loglog(y[0:50, :].T)
#         plt.show()



@exception2either
def fit_distribution(param):
    dist_fitter = DistFitter(param)
    dist_fitter.select_optimal()
    bstr_objs = dist_fitter.bootstrap()
    x_off, x_on = dist_fitter.minimize_rectangle(bstr_objs, 1e-4)
    #dist_fitter.plot1(bstr_objs)

    raw_max = np.max(dist_fitter.raw_data)
    raw_min = np.min(dist_fitter.raw_data)
    x_global = np.linspace(raw_min, raw_max, 100)
    opt_model = dist_fitter.optimal_model['model']
    #y_global = dist_fitter.optimal_model['model'].cdf(x_global)


    return {
        "distrib_name": dist_fitter.optimal_model['name'],
        "overall_params": dist_fitter.optimal_model['params'],
        "bootstaped_successes": len(bstr_objs),
        "raw": param["raw_data"],
        "distrib": {"x": x_global, "pdf": opt_model.pdf(x_global), "cdf": opt_model.cdf(x_global)},
        "x_off": x_off,
        "x_on": x_on
    }
