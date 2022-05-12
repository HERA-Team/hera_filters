# -*- coding: utf-8 -*-
# Copyright (c) 2018 The HERA Collaboration
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
from six.moves import range
from scipy.signal import windows
from warnings import warn
from scipy.optimize import leastsq, lsq_linear
import copy

#DEFAULT PARAMETERS FOR CLEANs
CLEAN_DEFAULTS_1D={'tol':1e-9, 'window':'none',
 'alpha':.5, 'maxiter':100, 'gain':0.1,
 'edgecut_low': 0, 'edgecut_hi': 0,
 'add_clean_residual':False, 'filt2d_mode':'rect'}
DAYENU_DEFAULTS_1D = {'suppression_factors' : [1e-9],
                 'max_contiguous_edge_flags' : 10}
DPSS_DEFAULTS_1D = {'suppression_factors' :  [1e-9],
               'eigenval_cutoff' : [1e-12],
               'max_contiguous_edge_flags' : 10}
DFT_DEFAULTS_1D = {'suppression_factors' : [1e-9],
                'fundamental_period' : np.nan,
                'max_contiguous_edge_flags' : 10}

CLEAN_DEFAULTS_2D = {'tol':1e-9, 'window': ['none', 'none'],
 'alpha':.5, 'maxiter':100, 'gain':0.1,
 'edgecut_low': [0, 0], 'edgecut_hi': [0, 0],
 'add_clean_residual':False, 'filt2d_mode':'rect'}
DAYENU_DEFAULTS_2D = {'suppression_factors' : [[1e-9], [1e-9]],
                 'max_contiguous_edge_flags' : 10}
DPSS_DEFAULTS_2D = {'suppression_factors' : [[1e-9], [1e-9]],
               'eigenval_cutoff' : [[1e-12], [1e-12]],
               'max_contiguous_edge_flags' : 10}
DFT_DEFAULTS_2D = {'suppression_factors' : [[1e-9], [1e-9]],
                'fundamental_period' : [np.nan, np.nan],
                'max_contiguous_edge_flags' : 10}


def _process_filter_kwargs(kwarg_dict, default_dict):
        """
        Utility function to complete a dictionary of kwargs
        by inserting values in default_dict for missing keys
        Also enforces correct spelling by making sure that all keys in
        kwarg_dict are in default_dict

        Parameters
        ----------
            kwarg_dict : dictionary of kwargs to process
            default_dict : dictionary containing default kwarg values.
        Returns
        -------
            Nothing, kwarg dict is corrected in place.
        """
        # check for mispelled keys
        for k in kwarg_dict:
            if not k in default_dict:
                raise ValueError("%s is not a valid argument!"%(k) + \
                                 "valid arguments include %s"%(list(default_dict.keys())))
        for k in default_dict:
            if not k in kwarg_dict:
                kwarg_dict[k] = default_dict[k]





def wedge_width(bl_len, sdf, nchan, standoff=0., horizon=1.):
    '''Return the (upper,lower) delay bins that geometrically correspond to the sky.
    Variable names preserved for backward compatability with capo/PAPER analysis.

    Arguments:
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        nchan: number of frequency channels
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered)
            Designed for area = np.ones(nchan, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    bl_dly = horizon * bl_len + standoff
    return calc_width(bl_dly, sdf, nchan)

def _get_filter_area(x, filter_center, filter_width):
    """
    Return an 'area' vector demarking where cleaning should be allowed
    to take place.

    Arguments:
        x : array-like real space vector listing where data is sampled.
        filter_center : center of the area to be cleaned. Units of 1/(x-units)
        filter_width : width of the region of area to be cleaned. Units of 1/(x-units)
    """
    nx = len(x)
    dx = np.mean(np.diff(x))
    if not np.isinf(filter_width):
        av = np.ones(len(x))
        filter_size = ((-filter_center + filter_width), (filter_center + filter_width))
        ut, lt = calc_width(filter_size, dx, nx)
        av[ut:lt] = 0.
    else:
        av = np.ones(nx)
    return av


def place_data_on_uniform_grid(x, data, weights, xtol=1e-3):
    """If possible, place data on a uniformly spaced grid.

    Given a vector of x-values (x), with data and weights,
    this function determines whether there are gaps in the
    provided x-values that are multiples of the minimum
    distance between x-values or whether any gaps are
    integer multiples of a fundamental grid spacing.
    If there are gaps that are integer multiples of a
    fundamental spacing, this function restores these
    x-values and inserts zero-valued
    data and zero-valued weights at their location,
    returning equally spaced data and weights that are
    effectively flagged at the missing x-values.
    This supports filtering data that was regularly sampled but has
    missing samples due to (for example) correlator dropouts since
    several of our filtering methods (DPSS fits and CLEAN) require data
    to be sampled on an equally spaced grid.

    Parameters
    ----------
    x: array-like,
        array of x-values.
    data: array-like,
        array of y-values.
        Should be the same length as x.
    weights: array-like,
        array of weights.
        Should be the same length as x.
    xtol: float, optional.
        fractional error tolerance to determine if x-values are
        on an incomplete grid.

    Returns
    -------
        xout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns x with all multiples of the fundamental unit filled in.
              If x is already uniformly spaced, returns x unchanged. If separations are not
              multiples of fundamental unit, also returns x unchanged.
        yout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns y with all multiples of the fundamental unit filled in with zeros.
              If x is already uniformly spaced, returns y unchanged. If separations are not
              multiples of fundamental unit, also returns y unchanged.
        wout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns w with all multiples of the fundamental unit filled in with zeros.
              If x is already uniformly spaced, returns w unchanged. If separations are not
              multiples of fundamental unit, also returns w unchanged.
        inserted: array-like
              boolean array indicating which x-values were inserted.
    """
    xdiff = np.diff(x)
    dx = np.abs(np.diff(x)).min() * np.sign(np.diff(x)[0])
    # first, check whether x, y, w already on a grid.
    # if they are, just return them.
    if np.allclose(xdiff, dx, rtol=0, atol=dx * xtol):
        xout = x
        dout = data
        wout = weights
        inserted = np.zeros(len(x), dtype=bool)
        return xout, dout, wout, inserted
    # next, check that the array is not on a grid and if it isn't, return x, y, w
    if not np.allclose(xdiff / dx, np.round(xdiff / dx), rtol=0.0, atol=np.abs(xtol * dx)):
        xout = x
        dout = data
        wout = weights
        inserted = np.zeros(len(x), dtype=bool)
        warn("Data cannot be placed on equally spaced grid! No values inserted.", RuntimeWarning)
        return xout, dout, wout, inserted
    # if the array is on a grid, then construct filled in grid.
    grid_size =int(np.round((x[-1] - x[0]) / dx)) + 1
    xout = np.linspace(x[0], x[-1], grid_size)
    dout = np.zeros(grid_size, dtype=np.complex128)
    wout = np.zeros(grid_size, dtype=np.float)
    inserted = np.ones(grid_size, dtype=bool)
    # fill in original data and weights.
    for x_index, xt in enumerate(x):
        output_index = np.argmin(np.abs(xout - xt))
        dout[output_index] = data[x_index]
        wout[output_index] = weights[x_index]
        inserted[output_index] = False

    return xout, dout, wout, inserted


def _fourier_filter_hash(filter_centers, filter_half_widths,
                         filter_factors, x, w=None, hash_decimal=10, **kwargs):
    '''
    Generate a hash key for a fourier filter

    Parameters
    ----------
        filter_centers: list,
                        list of floats for filter centers
        filter_half_widths: list
                        list of float filter half widths (in fourier space)

        filter_factors: list
                        list of float filter factors
        x: the x-axis of the data to be subjected to the hashed filter.
        w: optional vector of float weights to hash to. default, none
        hash_decimal: number of decimals to use for floats in key.
        kwargs: additional hashable elements the user would like to
                include in their filter key.

    Returns
    -------
    A key for fourier_filter arrays hasing the information provided in the args.
    '''
    filter_key = ('x:',) + tuple(np.round(x,hash_decimal))\
    + ('filter_centers x N x DF:',) + tuple(np.round(np.asarray(filter_centers) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + ('filter_half_widths x N x DF:',) + tuple(np.round(np.asarray(filter_half_widths) * np.mean(np.diff(x)) * len(x), hash_decimal))\
    + ('filter_factors x 1e9:',) + tuple(np.round(np.asarray(filter_factors) * 1e9, hash_decimal))
    if w is not None:
        filter_key = filter_key + ('weights', ) +  tuple(np.round(w.tolist(), hash_decimal))
    filter_key = filter_key + tuple([kwargs[k] for k in kwargs])
    return filter_key

def calc_width(filter_size, real_delta, nsamples):
    '''Calculate the upper and lower bin indices of a fourier filter

    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered

    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    if isinstance(filter_size, (list, tuple, np.ndarray)):
        _, l = calc_width(np.abs(filter_size[0]), real_delta, nsamples)
        u, _ = calc_width(np.abs(filter_size[1]), real_delta, nsamples)
        return (u, l)
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0:
        lthresh = nsamples
    return (uthresh, lthresh)

def fourier_filter(x, data, wgts, filter_centers, filter_half_widths, mode,
                   filter_dims=1, skip_wgt=0.1, zero_residual_flags=True, **filter_kwargs):
                   '''
                   A filtering function that wraps up all functionality of high_pass_fourier_filter
                   and add support for additional linear fitting options.
                   It can filter 1d or 2d data with x-axis(es) x and wgts in fourier domain
                   rectangular windows centered at filter_centers or filter_half_widths
                   perform filtering along any of 2 dimensions in 2d or 1d!
                   the 'dft' and 'dayenu' modes support irregularly sampled data.
                   Parameters
                   -----------
                   x: array-like
                      Array of floats giving x-values of data. Depending on the chosen method, this data may need to be equally spaced.
                      If performing a 2d clean, a 2-list or 2-tuple of np.ndarrays with x-values should be provided.
                    data: array-like
                        1d or 2d numpy.ndarray of complex data to filter.
                    wgts: array-like
                        1d or 2d numpy.ndarray of real weights. Must be the same shape as data.
                    filter_centers: array-like
                        if not 2dfilter: 1d np.ndarray or list or tuple of floats
                        specifying centers of rectangular fourier regions to filter.
                        If 2dfilter: should be a 2-list or 2-tuple. Each element
                        should be a list or tuple or np.ndarray of floats that include
                        centers of rectangular regions to filter.
                    filter_half_widths: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the half-widths of rectangular fourier regions to filter.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                    filter_dims, int or 2-list/tuple, optional
                        specify dimension to filter.
                        valid options are [0, 1, (1, 0)]
                    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                        time. Only works properly when all weights are all between 0 and 1.
                    mode: string
                        specify filtering mode. Currently supported are
                        'clean', iterative clean
                        'dpss_leastsq', dpss fitting using scipy.optimize.lsq_linear
                        'dft_leastsq', dft fitting using scipy.optimize.lsq_linear
                        'dpss_matrix', dpss fitting using direct lin-lsq matrix
                                       computation. Slower then lsq but provides linear
                                       operator that can be used to propagate
                                       statistics and the matrix is cached so
                                       on average, can be faster for data with
                                       many similar flagging patterns.
                        'dft_matrix', dft fitting using direct lin-lsq matrix
                                      computation. Slower then lsq but provides
                                      linear operator that can be used to propagate
                                      statistics and the matrix is cached so
                                      on average, can be faster for data with
                                      many similar flagging patterns.
                                      !!!WARNING: In my experience,
                                      'dft_matrix' option is numerical unstable.!!!
                                      'dpss_matrix' works much better.
                        'dayenu', apply dayenu filter to data. Does not
                                 deconvolve subtracted foregrounds.
                        'dayenu_dft_leastsq', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                'dft_leastsq' method (see above).
                        'dayenu_dpss_leastsq', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                 'dpss_leastsq' method (see above)
                        'dayenu_dft_matrix', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                'dft_matrix' mode (see above).
                                !!!WARNING: dft_matrix mode is often numerically
                                unstable. I don't recommend it!
                        'dayenu_dpss_matrix', apply dayenu filter to data
                                 and deconvolve subtracted foregrounds using
                                 'dpss_matrix' method (see above)
                        'dayenu_clean', apply dayenu filter to data. Deconvolve
                                 subtracted foregrounds with 'clean'.
                    zero_residual_flags : bool, optional.
                        If true, set flagged channels in the residual equal to zero.
                        Default is True.
                    filter_kwargs: additional arguments that are parsed as a dictionary
                        dictionary with options for fitting techniques.
                        if filter2d is true, this should be a 2-tuple or 2-list
                        of dictionaries. The dictionary for each dimension must
                        specify the following for each fitting method.
                        If mode=='dayenu', the user does not need to provide this argument.
                        * dft :
                            fundamental_period : float or 2-tuple
                                The fundamental_period of dft modes to fit. This is the
                                Fourier resolution of fitted fourier modes equal to
                                1/FP where FP is the fundamental period. For a standard
                                delay DFT FP = B where B is the visibility bandwidth
                                FP also sets the number of
                                modes fit within each window in 'filter_half_widths' will
                                equal fw / fundamental_period where fw is the filter width.
                                if filter2d, must provide a 2-tuple with fundamental_period
                                of each dimension.
                            suppression_factors : array-like
                                if len(filter_dims) == 1: 1d np.ndarray or list of tuples of floats
                                specifying the fractional residuals of model to leave in the data.
                                For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                                by the model.
                                if len(filter_dims) == 2: should be a 2-list or 2-tuple. Each element should
                                be a list or tuple or np.ndarray of floats that include centers
                                of rectangular bins.
                            max_contiguous_edge_flags  : int, optional
                                if the number of contiguous samples at the edge is greater then this
                                at either side, skip.
                        *  dayenu :
                            cache : dict, optional
                                dictionary for caching fitting matrices.
                            max_contiguous_edge_flags : int, optional
                                if the number of contiguous samples at the edge is greater then this
                                at either side, skip.
                        *  dpss :
                            eigenval_cutoff : array-like
                                list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                                if 2d filter, provide 2-tuple/list
                            suppression_factors : array-like
                                if not 2dfilter: 1d np.ndarray or list of tuples of floats
                                specifying the fractional residuals of model to leave in the data.
                                For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                                by the model.
                                if 2dfilter: should be a 2-list or 2-tuple. Each element should
                                be a list or tuple or np.ndarray of floats that include centers
                                of rectangular bins.
                                If mode == 'clean', this option can be left unspecified.'
                            max_contiguous_edge_flags : int, optional
                                if the number of contiguous samples at the edge is greater then this
                                at either side, skip.
                            cache : dict, optional
                                dictionary for caching fitting matrices.
                        * clean :
                             defaults can be accessed in dspec.CLEAN_DEFAULTS
                             tol : float,
                                clean tolerance. 1e-9 is standard.
                             maxiter : int
                                maximum number of clean iterations. 100 is standard.
                             filt2d_mode : string
                                if 'rect', clean withing a rectangular region of Fourier space given
                                by the intersection of each set of windows.
                                if 'plus' only clean the plus-shaped shape along
                                zero-delay and fringe rate.
                             edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                                be fed as a tuple specifying edgecut_low for first and second FFT axis.
                             edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                                be fed as a tuple specifying edgecut_hi for first and second FFT axis.
                             add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
                                in fourier space to the CLEAN model. Note that the residual actually returned is
                                not the CLEAN residual, but the residual in input data space.
                             window : window function for filtering applied to the filtered axis.
                                See dspec.gen_window for options. If clean2D, can be fed as a list
                                specifying the window for each axis in data.
                             gain : The fraction of a residual used in each iteration. If this is too low, clean takes
                                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
                             alpha : float, if window is 'tukey', this is its alpha parameter.

                    Returns
                    ---------
                        d_mdl: array-like
                            model -- best fit real-space model of data.
                        d_res: array-like
                            residual -- difference of data and model, nulled at flagged channels
                        info: dictionary with meta data on run and provided arguments.
                              clean uses a different info dict structure because of downstream code assumptions that are not
                              sufficiently general to describe the other methods. We should eventually migrate clean assumptions
                              to this format.
                              * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                                        - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                                                    of each step and values are a string that is either 'success' or 'skipped'.
                              * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                                        - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                                                    - 'mode': the filtering mode used to filter the time axis ('dayenu', 'dpss_leastsq' 'dpss_method')
                                                    - 'basis': (if using dpss/dft) gives the filtering basis.
                                                    - 'filter_centers': centers of filtering windows.
                                                    - 'filter_half_widths': half-widths of filtering regions for each axis.
                                                    - 'suppression_factors': amount of suppression for each filtering region.
                                                    - 'basis_options': the basis options used for dpss/dft mode. See dft_operator and dpss_operator for
                                                                       more details.
                                                    - 'x': vector of x-values used to generate the filter.
                            * 'clean_status': if CLEAN mode is used, this is also a field.
                                        - 'axis_0'/'axis_1': dictionary holding the clean output for cleaning on each axis. keys are integrations cleaned (integer)
                                                             and values for each key are the status dictionaries returned by aipy.deconv.clean (see aipy.deconv.clean
                                                             for more information).
                   '''
                   if not isinstance(filter_dims, (list, tuple)):
                       filter_dims = [filter_dims]
                   for d in filter_dims:
                       if not d in [0, 1, -1]:
                           raise ValueError("filter_dims can either contain 0, 1, or -1.")
                   supported_modes=['clean', 'dft_leastsq', 'dpss_leastsq', 'dft_matrix', 'dpss_matrix', 'dayenu',
                                    'dayenu_dft_leastsq', 'dayenu_dpss_leastsq', 'dayenu_dpss_matrix',
                                    'dayenu_dft_matrix', 'dayenu_clean']
                   if not mode in supported_modes:
                       raise ValueError("Need to supply a mode in supported modes:%s"%(str(supported_modes)))
                   mode = mode.split('_')
                   ndim_data = data.ndim
                   ndim_wgts = wgts.ndim
                   if not (ndim_data == 1 or ndim_data == 2):
                       raise ValueError("data must be a 1D or 2D ndarray")
                   if not ndim_wgts == ndim_data:
                       raise ValueError("Number of dimensions in weights, %d does not equal number of dimensions in data, %d!"%(ndim_wgts, ndim_data))
                   #The core code of this method will always assume 2d data
                   if ndim_data == 1:
                       data = np.asarray([data])
                       wgts = np.asarray([wgts])
                   if len(filter_dims) == 2:
                       filter2d = True
                   elif len(filter_dims) == 1:
                       filter2d = False
                   else:
                     raise ValueError("filter_dims either have length 1 or length 2")
                   if 0 in filter_dims and not filter2d:
                     # If we are only filtering along the time-axis
                     # then we set data and weights equal to their transposes
                     # and proceed to filter as though we are filtering across
                     # the frequency axis.
                     # the transposes are undone below after filtering is complete
                     data = data.T
                     wgts = wgts.T
                   if 'cache' not in filter_kwargs:
                      cache = {}
                   else:
                      cache = filter_kwargs.pop('cache')
                   # process filter_kwargs
                   if 'dayenu' == mode[0]:
                       if len(mode) > 1:
                           if 'dft' in mode:
                               if filter2d:
                                   defaults = {**DAYENU_DEFAULTS_2D, **DFT_DEFAULTS_2D}
                               else:
                                   defaults = {**DAYENU_DEFAULTS_1D, **DFT_DEFAULTS_1D}
                           elif 'dpss' in mode:
                               if filter2d:
                                   defaults = {**DAYENU_DEFAULTS_2D, **DPSS_DEFAULTS_2D}
                               else:
                                   defaults = {**DAYENU_DEFAULTS_1D, **DPSS_DEFAULTS_1D}
                       else:
                           if filter2d:
                               defaults = DAYENU_DEFAULTS_2D
                           else:
                               defaults = DAYENU_DEFAULTS_1D
                   elif mode[0] == 'dft':
                       if filter2d:
                           defaults = DFT_DEFAULTS_2D
                       else:
                           defaults = DFT_DEFAULTS_1D
                   elif mode[0] == 'dpss':
                       if filter2d:
                           defaults = DPSS_DEFAULTS_2D
                       else:
                           defaults = DPSS_DEFAULTS_1D
                   elif mode[0] == 'clean':
                       if filter2d:
                           defaults = CLEAN_DEFAULTS_2D
                       else:
                           defaults = CLEAN_DEFAULTS_1D

                   _process_filter_kwargs(filter_kwargs, defaults)
                   if 'dft' in mode:
                        fp = np.asarray(filter_kwargs['fundamental_period']).flatten()
                        if filter2d:
                            for m in range(len(fp)):
                                if np.isnan(fp[m]):
                                    fp[m] = 2. * (x[m].max() - x[m].min())
                        else:
                            if np.isnan(fp[0]):
                                fp = [2. * (x.max() - x.min())]

                        if len(fp) == 1:
                            filter_kwargs['fundamental_period'] = fp[0]
                        else:
                            filter_kwargs['fundamental_period'] = list(fp)

                   if mode[0] == 'dayenu':
                       if zero_residual_flags is None:
                           zero_residual_flags = True
                       if filter2d:
                          filter_dims_d = [1, 0]
                       else:
                          # If filter_dimes = [0], then the data and wgts have already been transposed
                          # so that the 1d filtering is executed as though we are filtering in frequency
                          # the transposes are undone below (after filtering)
                          filter_dims_d = [1]
                       suppression_factors = filter_kwargs.pop('suppression_factors')
                       max_contiguous_edge_flags = filter_kwargs.pop('max_contiguous_edge_flags')
                       residual, info = dayenu_filter(x=x, data=data, wgts=wgts, filter_dimensions=filter_dims_d,
                                                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                                     filter_factors=suppression_factors, cache=cache, skip_wgt=skip_wgt,
                                                     max_contiguous_edge_flags=max_contiguous_edge_flags,
                                                     zero_residual_flags=zero_residual_flags)
                       model = data - residual
                       if len(mode) > 1:
                           model, _, info_deconv = _fit_basis_2d(x=x, data=model, filter_centers=filter_centers, filter_dims=filter_dims_d,
                                                                 skip_wgt=skip_wgt, basis=mode[1], method=mode[2], wgts=wgts, basis_options=filter_kwargs,
                                                                 filter_half_widths=filter_half_widths, suppression_factors=suppression_factors,
                                                                 cache=cache, max_contiguous_edge_flags=max_contiguous_edge_flags,
                                                                 zero_residual_flags=zero_residual_flags)
                           info['info_deconv']=info_deconv

                   elif mode[0] in ['dft', 'dpss']:
                       if zero_residual_flags is None:
                           zero_residual_flags = True
                       if filter2d:
                           filter_dims_d = [1, 0]
                       else:
                           # If filter_dimes = [0], then the data and wgts have already been transposed
                           # so that the 1d filtering is executed as though we are filtering in frequency
                           # the transposes are undone below (after filtering)
                           filter_dims_d = [1]
                       suppression_factors = filter_kwargs.pop('suppression_factors')
                       max_contiguous_edge_flags = filter_kwargs.pop('max_contiguous_edge_flags')
                       #if filter2d is True, create fitting_options that is a 2-list for 0 and 1 dimension
                       model, residual, info = _fit_basis_2d(x=x, data=data, filter_centers=filter_centers, filter_dims=filter_dims_d,
                                                           skip_wgt=skip_wgt, basis=mode[0], method=mode[1], wgts=wgts, basis_options=filter_kwargs,
                                                           filter_half_widths=filter_half_widths, suppression_factors=suppression_factors,
                                                           cache=cache, max_contiguous_edge_flags=max_contiguous_edge_flags,
                                                           zero_residual_flags=zero_residual_flags)
                   elif mode[0] == 'clean':
                       if zero_residual_flags is None:
                           zero_residual_flags = False
                       model, residual, info = _clean_filter(x=x, data=data, wgts=wgts, filter_centers=filter_centers, skip_wgt=skip_wgt,
                                                            filter_half_widths=filter_half_widths, clean2d=filter2d, zero_residual_flags=zero_residual_flags,
                                                             **filter_kwargs)
                       if filter2d:
                           info['filter_params']['axis_0'] = filter_kwargs
                           info['filter_params']['axis_1'] = info['filter_params']['axis_0']
                       else:
                           info['filter_params']['axis_1'] = filter_kwargs
                   if 0 in filter_dims and not filter2d:
                        # undo transposes if we were performing a dimension 0
                        # time filter.
                        model = model.T
                        residual = residual.T
                        data = data.T
                        wgts = wgts.T
                        # switch axis 0 and axis 1 info dicts if we were doing time cleaning.
                        for k in info:
                            if not k == 'info_deconv':
                                info[k]['axis_0'] = copy.deepcopy(info[k]['axis_1'])
                                info[k]['axis_1'] = {}
                        # if we deconvolve the subtracted foregrounds in dayenu
                        # then provide fitting options for the deconvolution.
                        if 'info_deconv' in info:
                            for k in info['info_deconv']:
                                info['info_deconv'][k]['axis_0'] = copy.deepcopy(info['info_deconv'][k]['axis_1'])
                                info['info_deconv'][k]['axis_1'] = {}
                   if ndim_data == 1:
                       model = model.flatten()
                       residual = residual.flatten()
                   return model, residual, info

def vis_clean(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False):
        '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. Default is a 1D clean
        on the last axis of data.

        Arguments:
            data: 1D or 2D (real or complex) numpy array to be filtered.
                (Unlike previous versions, it is NOT assumed that weights have already been multiplied
                into the data.)
            wgts: real numpy array of linear multiplicative weights with the same shape as the data.
            filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
                space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
                Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
                and positive bound of the filter in fourier space respectively.
                Example: (20, 40) --> (-20 < tau < 40)
             real_delta: the bin width in real space of the dimension to be filtered.
                If 2D cleaning, then real_delta must also be a len-2 list.
            clean2d : bool, if True perform 2D clean, else perform a 1D clean on last axis.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis.
                See dspec.gen_window for options. If clean2D, can be fed as a list
                specifying the window for each axis in data.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                time. Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            gain: The fraction of a residual used in each iteration. If this is too low, clean takes
                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
            alpha : float, if window is 'tukey', this is its alpha parameter.
            filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
                If 'rect', a 2D rectangular filter is constructed in fourier space (default).
                If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
                slice along 0 delay and fringe-rate is kept.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                be fed as a tuple specifying edgecut_low for first and second FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. For 2D cleaning, can
                be fed as a tuple specifying edgecut_hi for first and second FFT axis.
            add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
                in fourier space to the CLEAN model. Note that the residual actually returned is
                not the CLEAN residual, but the residual in input data space.
        Returns:
            d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
            d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
            info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
        '''
        if clean2d:
            filter_dims = [1, 0]
            if not isinstance(real_delta, (list, tuple)) or not len(real_delta) == 2:
                raise ValueError("real_delta must be length 2 list or tuple if clean2d is True.")
            x = [np.arange(data.shape[0]) * real_delta[0],
                 np.arange(data.shape[1]) * real_delta[1]]
            fc = []
            fw = []
            for m in range(2):
                if isinstance(filter_size[m], (list, tuple)):
                    fc.append([(filter_size[m][1] - filter_size[m][0]) / 2.])
                    fw.append([np.abs(filter_size[m][1] + filter_size[m][0]) / 2.])
                else:
                    fc.append([0.])
                    fw.append([filter_size[m]])
            edgecut_hi = (edgecut_hi, edgecut_hi)
            edgecut_low = (edgecut_low, edgecut_low)
            window = (window, window)

        else:
            filter_dims = 1
            x = np.arange(data.shape[-1]) * real_delta
            if isinstance(filter_size, (list, tuple)):
                fc = [(filter_size[1] - filter_size[0]) / 2.]
                fw = [np.abs(filter_size[1] + filter_size[0]) / 2.]
            else:
                fc = [0.]
                fw = [filter_size]
        return fourier_filter(x, data, wgts, filter_centers=fc, filter_half_widths=fw,
                              mode='clean', filter_dims=filter_dims, tol=tol, window=window,
                              maxiter=maxiter, gain=gain, filt2d_mode=filt2d_mode, alpha=alpha,
                              edgecut_hi=edgecut_hi, edgecut_low=edgecut_low, add_clean_residual=add_clean_residual)


#TODO: Add DPSS interpolation function to this.
def high_pass_fourier_filter(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False):
    '''
    wrapper for vis_clean to support backwards compatibility. See vis_clean docstring.
    '''
    return vis_clean(data=data, wgts=wgts, filter_size=filter_size, real_delta=real_delta, clean2d=clean2d,
                     tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, filt2d_mode=filt2d_mode,
                     alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, add_clean_residual=add_clean_residual)

def dayenu_filter(x, data, wgts, filter_dimensions, filter_centers, filter_half_widths, filter_factors,
                  cache = {}, return_matrices=True, hash_decimal=10, skip_wgt=0.1, max_contiguous_edge_flags=10,
                  zero_residual_flags=True):
    '''
    Apply a linear delay filter to waterfall data.
    Due to performance reasons, linear filtering only supports separable delay/fringe-rate filters.

    Arguments
    ---------
    x: array-like or length-2 list/tuples that are array-like
        x-values for each data point in dimension to be filtered.
    data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
    Does not assume that weights have already been multiplied!
    wgts: real numpy array of linear multiplicative weights with the same shape as the data.
    filter_dimensions: list
        list of integers indicating data dimensions to filter. Must be 0, 1, or -1
    filter_centers: float, list, or 1d numpy array of delays at which to center filter windows
        Typically in units of (seconds)
    filter_half_widths: float, list, or 1d numpy array of half-widths of each delay filtere window
        with centers specified by filter_centers.
        Typically in units of (seconds)
    filter_factors: float, list, or 1d numpy array of factors by which filtering should be
        applied within each filter window specified in filter_centers and
        filter_half_widths. If a float or length-1 list/ndarray is provided,
        the same filter factor will be used in every filter window.
    cache: optional dictionary for storing pre-computed delay filter matrices.
    return_matrices: bool,
        if True, return a dict referencing every every filtering matrix used.
    hash_decimal: number of decimals to hash x to
    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
        time. Only works properly when all weights are all between 0 and 1.
    max_contiguous_edge_flags : int, optional
        if the number of contiguous samples at the edge is greater then this
        at either side, skip .
    zero_residual_flags : bool, optional.
        If true, set flagged channels in the residual equal to zero.
        Default is True.
    Returns
    -------
    data: array, 2d clean residual with data filtered along the frequency direction.
    info: dictionary with filtering parameters and a list of skipped_times and skipped_channels
          has the following fields
         * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                   - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                               of each step and values are a string that is either 'success' or 'skipped'.
         * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                   - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                               - 'filter_centers': centers of filtering windows.
                               - 'filter_half_widths': half-widths of filtering regions for each axis.
                               - 'suppression_factors': amount of suppression for each filtering region.
                               - 'x': vector of x-values used to generate the filter.
    '''
    # check that data and weight shapes are consistent.
    d_shape = data.shape
    w_shape = wgts.shape
    d_dim = data.ndim
    w_dim = wgts.ndim
    if not (d_dim == 1 or d_dim == 2):
        raise ValueError("number of dimensions in data array does not "
                         "equal 1 or 2! data dim = %d"%(d_dim))
    if not (w_dim == 1 or w_dim == 2):
        raise ValueError("number of dimensions in wgts array does not "
                         "equal 1 or 2! wght dim = %d"%(w_dim))
    if not w_dim == d_dim:
        raise ValueError("number of dimensions in data array does not equal "
                         "number of dimensions in weights array."
                         "data.dim == %d, wgts.dim == %d"%(d_dim, w_dim))
    for dim in range(d_dim):
        if not d_shape[dim] == w_shape[dim]:
            raise ValueError("number of elements along data dimension %d, nel=%d"
                             "does not equal the number of elements along weight"
                             "dimension %d, nel = %d"%(dim, d_shape[dim], dim, w_shape[dim]))
    if not isinstance(x, (np.ndarray,list, tuple)):
        raise ValueError("x must be a numpy array, list, or tuple")
    # Check that inputs are tiples or lists
    if not isinstance(filter_dimensions, (list,tuple,int, np.int)):
        raise ValueError("filter_dimensions must be a list or tuple")
    # if filter_dimensions are supplied as a single integer, convert to list (core code assumes lists).
    if isinstance(filter_dimensions, int):
        filter_dimensions = [filter_dimensions]
    # check that filter_dimensions is no longer then 2 elements
    if not len(filter_dimensions) in [1, 2]:
        raise ValueError("length of filter_dimensions cannot exceed 2")
    # make sure filter_dimensions are 0 or 1.
    for dim in filter_dimensions:
        if not dim in [0, 1] or not isinstance(dim, (int, np.int)):
            raise ValueError("filter dimension must be integer 0, or 1")

    # convert filter dimensions to a list of integers (incase the dimensions were supplied as floats)
    # will only filter each dim a single time.
    # now check validity of other inputs. We perform the same check over multiple
    # inputs by iterating over a list with their names.
    #convert 1d data to 2d data to save lines of code.
    if d_dim == 1:
        data = np.asarray([data])
        wgts = np.asarray([wgts])
        data_1d = True
        # 1d data will result in nonsensical filtering along zeroth axis.
        filter_dimensions=[1]
    else:
        data_1d = False
    nchan = data.shape[1]
    ntimes = data.shape[0]
    check_vars = [filter_centers, filter_half_widths, filter_factors]
    check_names = ['filter_centers', 'filter_half_widths', 'filter_factors']
    for anum, aname, avar in zip(range(len(check_vars)),check_names,check_vars):
        # If any of these inputs is a float or numpy array, convert to a list.
        if isinstance(avar, np.ndarray):
            check_vars[anum] = list(avar)
        elif isinstance(avar, np.float):
            check_vars[anum] = [avar]

    filter_centers,filter_half_widths,filter_factors = check_vars
    # Next, perform some checks that depend on the filtering dimensions provided.
    if 0 in filter_dimensions and 1 in filter_dimensions:
        for avar,aname in zip(check_vars,check_names):
            err_msg = "2d clean specified! %s must be a length-2 list of lists for 2d clean"%aname
            # if we are going to filter in dimension 1 and 0, make sure that each input
            # listed in check_vars is a length-2 list of lists.
            if len(avar) == 2:
                if not (isinstance(avar[0], list) and isinstance(avar[1], list)):
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
        if not len(x) == 2:
            raise ValueError("For 2d filtering, x must be 2d long list or tuple or ndarray")
        for j in range(2):
            if not isinstance(x[j], (tuple, list, np.ndarray)):
                raise ValueError("x[%d] must be a tuple, list or numpy array."%(j))
            x[j]=np.asarray(x[j])
        for ff_num,ff_list in zip(filter_dimensions,filter_factors):
            # we allow the user to provide a single filter factor for multiple
            # filtering windows on a single dimension. This code
            # iterates through each dimension and if a single filter_factor is provided
            # it converts the filter_factor list to a list of filter_factors with the same
            # length as filter_centers.
            if len(ff_list) == 1:
                ff_list = [ff_list[0] for m in range(len(filter_centers[ff_num]))]


    else:
        if len(filter_factors) == 1:
            # extend filter factor list of user supplied a float or len-1 list.
            filter_factors = [filter_factors[0] for m in range(len(filter_centers))]
        if 0 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [filter_factors,[]]
            filter_centers = [filter_centers,[]]
            filter_half_widths = [filter_half_widths,[]]
            x = [x,None]
        elif 1 in filter_dimensions:
            # convert 1d input-lists to
            # a list of lists for core-code to operate on.
            filter_factors = [[],filter_factors]
            filter_centers = [[],filter_centers]
            filter_half_widths = [[],filter_half_widths]
            x = [None, x]
    check_vars = [filter_centers, filter_half_widths, filter_factors]
    # Now check that the number of filter factors = number of filter widths
    # = number of filter centers for each dimension.
    for fs in filter_dimensions:
        for aname1,avar1 in zip(check_names,check_vars):
            for aname2,avar2 in zip(check_names,check_vars):
                if not len(avar1[fs]) == len(avar2[fs]):
                    raise ValueError("Number of elements in %s-%d must equal the"
                                     " number of elements %s-%d!"%(aname1, fs, aname2, fs))

    info = {'status':{'axis_0':{}, 'axis_1':{}}, 'filter_params':{'axis_0':{}, 'axis_1':{}}}
    for fs in range(2):
        info['filter_params']['axis_%d'%fs]['filter_centers'] = filter_centers[fs]
        info['filter_params']['axis_%d'%fs]['filter_half_widths'] = filter_half_widths[fs]
        info['filter_params']['axis_%d'%fs]['filter_factors'] = filter_factors[fs]
        info['filter_params']['axis_%d'%fs]['x'] = x[fs]
        info['filter_params']['axis_%d'%fs]['mode'] = 'dayenu'
    skipped = [[],[]]
    # in the lines below, we iterate over the time dimension. For each time, we
    # compute a lazy covariance matrix (filter_mat) from the weights (wght) and
    # a sinc downweight matrix. (dayenu_mat_inv). We then attempt to
    # take the psuedo inverse to get a filtering matrix that removes foregrounds.
    # we do this for the zeroth and first filter dimension.
    output = copy.deepcopy(data)
    #this loop iterates through dimensions to iterate over (fs is the non-filter
    #axis).
    filter_matrices=[{},{}]
    #check filter factors for zeros and negative numbers
    for ff in filter_factors:
        for fv in ff:
            if fv <= 0.:
                raise ValueError("All filter factors must be greater than zero! You provided %.2e :(!"%(fv))

    for fs in filter_dimensions:
        if fs == 0:
            _d, _w = output.T, wgts.T
        else:
            _d, _w = output, wgts
        #if the axis orthogonal to the iteration axis is to be filtered, then
        #filter it!.
        for sample_num, sample, wght in zip(range(data.shape[fs-1]), _d, _w):
            filter_key = _fourier_filter_hash(filter_centers=filter_centers[fs], filter_half_widths=filter_half_widths[fs],
                                              filter_factors=filter_factors[fs], x=x[fs], w=wght,
                                              label='dayenu_filter_matrix')
            if not filter_key in cache:
                #only calculate filter matrix and psuedo-inverse explicitly if they are not already cached
                #(saves calculation time).
                if np.count_nonzero(wght) / len(wght) >= skip_wgt and np.count_nonzero(wght[:max_contiguous_edge_flags]) > 0 \
                   and np.count_nonzero(wght[-max_contiguous_edge_flags:]) >0:
                    wght_mat = np.outer(wght.T, wght)
                    filter_mat = dayenu_mat_inv(x=x[fs], filter_centers=filter_centers[fs],
                                                         filter_half_widths=filter_half_widths[fs],
                                                         filter_factors=filter_factors[fs], cache=cache) * wght_mat
                    try:
                        #Try taking psuedo-inverse. Occasionally I've encountered SVD errors
                        #when a lot of channels are flagged. Interestingly enough, I haven't
                        #I'm not sure what the precise conditions for the error are but
                        #I'm catching it here.
                        cache[filter_key] = np.linalg.pinv(filter_mat)
                    except np.linalg.LinAlgError:
                        # skip if we can't invert or psuedo-invert the matrix.
                        cache[filter_key] = None
                else:
                    # skip if we don't meet skip_wegith criterion or continuous edge flags
                    # are to many. This last item isn't really a problem for dayenu
                    # but it's here for consistancy.
                    cache[filter_key] = None

            filter_mat = cache[filter_key]
            if filter_mat is not None:
                if fs == 0:
                    output[:, sample_num] = np.dot(filter_mat, sample)
                elif fs == 1:
                    output[sample_num] = np.dot(filter_mat, sample)
                info['status']['axis_%d'%fs][sample_num] = 'success'
            else:
                skipped[fs-1].append(sample_num)
                info['status']['axis_%d'%fs][sample_num] = 'skipped'
            if return_matrices:
                filter_matrices[fs][sample_num]=filter_mat
    if zero_residual_flags:
        output = output * (~np.isclose(wgts, 0., atol=1e-10)).astype(float)
    # set residual equal to zero where weights are zero.
    #1d data will only be filtered across "channels".
    if data_1d and ntimes == 1:
        output = output[0]
    return output, info


def wedge_filter(data, wgts, bl_len, sdf, standoff=0., horizon=1., min_dly=0.0, skip_wgt=0.5,
                 mode='clean', **kwargs):
    '''Apply a wideband delay filter to data. Variable names preserved for
        backward compatability with capo/PAPER analysis.

    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        min_dly: a minimum delay used for cleaning: if bl_dly < min_dly, use min_dly. same units as bl_len
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        mode: filtering mode (see supported modes in fourier_filter docstring)
        kwargs: see fourier_filter documentation
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # get bl delay
    bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)
    return delay_filter(sdf=sdf, data=data, wgts=wgts, max_dly=bl_dly,
                          skip_wgt=skip_wgt, **kwargs)

def delay_filter(data, wgts, max_dly, sdf, skip_wgt=0.5,
                 mode='clean', **kwargs):
    '''Apply a wideband delay filter to data. Variable names preserved for
        backward compatability with capo/PAPER analysis.

    Arguments:
        data: 1D or 2D (real or complex) numpy array where last dimension is frequency.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
            max_dly: maximum abs of delay to filter to (around delay = 0.)
        sdf: frequency channel width (typically in GHz)
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        mode: filtering mode (see supported modes in fourier_filter docstring)
        kwargs: see fourier_filter documentation
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    freqs = np.arange(data.shape[-1]) * sdf
    return fourier_filter(x=freqs, data=data, wgts=wgts, filter_centers=[0.], filter_half_widths=[max_dly],
                          skip_wgt=skip_wgt, filter_dims=1, mode=mode, **kwargs)

def fringe_filter(data, wgts, max_frate, dt, skip_wgt=0.5, mode='clean', **kwargs):
    """
    Run a CLEAN deconvolution along the time axis.

    Args:
        data : 1D or 2D data array. If 2D, shape=(Ntimes, Nfreqs)
        wgts : 1D or 2D weight array.
        max_frate : float, maximum fringe-rate (i.e. frequency) to CLEAN, units of 1/[dt]. 2-tuple
        can be provided where (20, 20) would clean between -20 < fr < 20
        dt : float, time-bin width of data
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        mode: filtering mode (see supported modes in fourier_filter docstring)
        kwargs: see fourier_filter docstring.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    """
    # print deprecation warning
    times = np.arange(data.shape[0]) * dt
    # run fourier filter
    if isinstance(max_frate, (list,tuple)):
        fc = (max_frate[1] - max_frate[0]) / 2.
        fw = np.abs(max_frate[1] + max_frate[0]) / 2.
    else:
        fc = 0.
        fw = max_frate
    return fourier_filter(x=times, data=data, wgts=wgts, filter_centers=[fc], filter_half_widths=[fw],
                          skip_wgt=skip_wgt, filter_dims=0, mode=mode, **kwargs)

def vis_filter(data, wgts, max_frate=None, dt=None, bl_len=None, sdf=None, standoff=0.0, horizon=1., min_dly=0.,
               tol=1e-4, window='none', maxiter=100, gain=1e-1, skip_wgt=0.5, filt2d_mode='rect',
               edgecut_low=0, edgecut_hi=0, alpha=0.5, add_clean_residual=False):
    """
    A generalized interface to delay and/or fringe-rate 1D CLEAN functions, or a full 2D clean
    if both bl_len & sdf and max_frate & dt variables are specified.

    Args:
        data : 1D or 2D data array. If 2D has shape=(Ntimes, Nfreqs)
        wgts : float weight array, matching shape of data
        max_frate : float, maximum fringe-rate (i.e. frequency) to CLEAN, units of 1/[dt]
        dt : float, time-bin width [sec]
        bl_len: length of baseline (in 1/[sdf], typically ns)
        sdf: frequency channel width (typically in GHz)
        standoff: fixed additional delay beyond the horizon (same units as bl_len)
        horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
        min_dly: a minimum delay used for cleaning: if bl_dly < min_dly, use min_dly. same units as bl_len
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis.
            See gen_window for options.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
            If 'rect', a 2D rectangular filter is constructed in fourier space (default).
            If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
            slice along 0 delay and fringe-rate is kept.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        alpha : float, if window is tukey, this is its alpha parameter.
        add_clean_residual : bool, if True, adds the residual within the CLEAN bounds
            in fourier space to the CLEAN model (and sets residual within CLEAN bounds to zero).
            This is more in-line with a standard filtering operation, rather than a CLEAN operation.
            If False, residual is not added to the CLEAN model.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    """
    # print deprecation warning
    warn("Warning: dspec.vis_filter will soon be deprecated in favor of filtering.vis_filter",
         DeprecationWarning)

    # type checks
    timeclean = False
    if dt is not None or max_frate is not None:
        timeclean = True
        assert max_frate is not None and dt is not None, "Must specify both max_frate and dt for time cleaning"

    freqclean = False
    if sdf is not None or bl_len is not None:
        freqclean = True
        assert sdf is not None and bl_len is not None, "Must specify both bl_len and sdf for frequency cleaning"

    clean2d = timeclean and freqclean

    # 1D clean
    if not clean2d:
        # time clean
        if timeclean:
            mdl, res, info = fringe_filter(data, wgts, max_frate, dt=dt, tol=tol, window=window, edgecut_low=edgecut_low,
                                                      edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter,
                                                      gain=gain, alpha=alpha, mode='clean')
        # freq clean
        elif freqclean:
            mdl, res, info = wedge_filter(data, wgts, standoff=standoff, horizon=horizon, sdf=sdf, tol=tol,
                                          window=window, edgecut_low=edgecut_low, bl_len=bl_len,
                                          edgecut_hi=edgecut_hi, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain,
                                          alpha=alpha, mode='clean')

    # 2D clean
    else:
        # get bl delay
        bl_dly = _get_bl_dly(bl_len, horizon=horizon, standoff=standoff, min_dly=min_dly)
        freqs = np.arange(data.shape[1]) * sdf
        times = np.arange(data.shape[0]) * dt
        if isinstance(max_frate, (list,tuple)):
            fc = (max_frate[1] - max_frate[0]) / 2.
            fw = np.abs(max_frate[1] + max_frate[0]) / 2.
        else:
            fc = 0.
            fw = max_frate
        # 2D clean
        if isinstance(edgecut_hi, (int, np.int)):
            edgecut_hi = (edgecut_hi, edgecut_hi)
        if isinstance(edgecut_low, (int, np.int)):
            edgecut_low = (edgecut_low, edgecut_low)
        if isinstance(window, str):
            window = (window, window)
        mdl, res, info = fourier_filter(data=data, wgts=wgts, x=[times, freqs], filter_centers=[[fc], [0.]],
                                        filter_half_widths=[[fw], [bl_dly]], filter_dims=[1, 0], tol=tol,
                                        window=window, edgecut_low=edgecut_low,
                                        edgecut_hi=edgecut_hi, maxiter=maxiter,
                                        gain=gain, filt2d_mode=filt2d_mode,
                                        alpha=alpha, add_clean_residual=add_clean_residual, mode='clean')

    return mdl, res, info


def _get_bl_dly(bl_len, horizon=1., standoff=0., min_dly=0.):
    # construct baseline delay
    bl_dly = horizon * bl_len + standoff

    # check minimum delay
    bl_dly = np.max([bl_dly, min_dly])

    return bl_dly


def gen_window(window, N, alpha=0.5, edgecut_low=0, edgecut_hi=0, normalization=None, **kwargs):
    """
    Generate a 1D window function of length N.

    Args:
        window : str, window function
        N : int, number of channels for windowing function.
        edgecut_low : int, number of bins to consider as zero-padded at the low-side
            of the array, such that the window smoothly connects to zero.
        edgecut_hi : int, number of bins to consider as zero-padded at the high-side
            of the array, such that the window smoothly connects to zero.
        alpha : if window is 'tukey', this is its alpha parameter.
        normalization : str, optional
            set to 'rms' to divide by rms and 'mean' to divide by mean.
    """
    if normalization is not None:
        if normalization not in ["mean", "rms"]:
            raise ValueError("normalization must be one of ['rms', 'mean']")
    # parse multiple input window or special windows
    w = np.zeros(N, dtype=np.float)
    Ncut = edgecut_low + edgecut_hi
    if Ncut >= N:
        raise ValueError("Ncut >= N for edgecut_low {} and edgecut_hi {}".format(edgecut_low, edgecut_hi))
    if edgecut_hi > 0:
        edgecut_hi = -edgecut_hi
    else:
        edgecut_hi = None
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w[edgecut_low:edgecut_hi] = windows.boxcar(N - Ncut)
    elif window in ['blackmanharris', 'blackman-harris', 'bh', 'bh4']:
        w[edgecut_low:edgecut_hi] =  windows.blackmanharris(N - Ncut)
    elif window in ['hanning', 'hann']:
        w[edgecut_low:edgecut_hi] =  windows.hann(N - Ncut)
    elif window == 'tukey':
        w[edgecut_low:edgecut_hi] =  windows.tukey(N - Ncut, alpha)
    elif window in ['blackmanharris-7term', 'blackman-harris-7term', 'bh7']:
        # https://ieeexplore.ieee.org/document/293419
        a_k = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837,
              0.00077658482522, 0.00001388721735]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-9term', 'cosinesum9term', 'cs9']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.384331152777942e-1, 4.00554534864382e-1, 2.358242530472107e-1, 9.527918858383112e-2,
               2.537395516617152e-2, 4.152432907505835e-3, 3.68560416329818e-4, 1.38435559391703e-5,
               1.161808358932861e-7]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-11term', 'cosinesum11term', 'cs11']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.151527506679809e-1, 3.731348357785249e-1, 2.424243358446660e-1, 1.166907592689211e-1,
               4.077422105878731e-2, 1.000904500852923e-2, 1.639806917362033e-3, 1.651660820997142e-4,
               8.884663168541479e-6, 1.938617116029048e-7, 8.482485599330470e-10]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    else:
        try:
            # return any single-arg window from windows
            w[edgecut_low:edgecut_hi] = getattr(windows, window)(N - Ncut)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))
    if normalization == 'rms':
        w /= np.sqrt(np.mean(np.abs(w)**2.))
    if normalization == 'mean':
        w /= w.mean()
    return w


def fourier_operator(dsize, nmax, nmin=None, L=None):
    """
    Return a complex Fourier analysis operator for a given data dimension and number of Fourier modes.

    Parameters
    ----------
    dsize : int
        Size of data array.

    nmax : int
        Maximum Fourier mode number. Modes will be constructed between
        [nmin, nmax], for a total of (nmax - min) + 1 modes.
    nmin : int, optional, default nmin = nmax
        minimum integer of fourier mode numbers. Modes will be constructed between
        [nmin, nmax] for total of (nmax - nmin) + 1 modes.
    L : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.
    Returns
    -------
    F : array_like
        Fourier matrix operator, of shape (Nmodes, Ndata)
    """
    nu = np.arange(dsize)
    if L is None:
        L = nu[-1] - nu[0]
    if nmin is None:
        nmin = -nmax
    # Construct frequency array (*not* in physical frequency units)
    # Build matrix operator for complex Fourier basis
    n = np.arange(nmin, nmax + 1)
    F = np.array([np.exp(-1.j * _n * nu / L) for _n in n])
    return F


def fourier_model(cn, Nfreqs):
    """
    Calculate a 1D (complex) Fourier series model from a set of complex coefficients.

    Parameters
    ----------
    coeffs : array_like
        Array of complex Fourier coefficients, ordered from (-n, n), where n is
        the highest harmonic mode in the model.

    Nfreqs : int
        Number of frequency channels to model. The Fourier modes are integer
        harmonics within this frequency window.

    Returns
    -------
    model : array_like
        Fourier model constructed from the input harmonic coefficients.
        Shape: (Nfreqs,).
    """
    try:
        cn_shape = cn.shape
    except AttributeError:
        raise ValueError("cn must be a 1D array")
    if len(cn.shape) != 1:
        raise ValueError("cn must be a 1D array")
    nmax = (cn.size - 1) // 2  # Max. harmonic

    # Build matrix operator for complex Fourier basis
    F = fourier_operator(dsize=Nfreqs, nmax=nmax)

    # Return model
    return np.dot(cn, F)


def delay_filter_leastsq_1d(data, flags, sigma, nmax, add_noise=False,
                            cn_guess=None, use_linear=True, operator=None, fundamental_period=None):
    """
    Fit a smooth model to 1D complex-valued data with flags, using a linear
    least-squares solver. The model is a Fourier series up to a specified
    order. As well as calculating a best-fit model, this will also return a
    copy of the data with flagged regions filled in ('in-painted') with the
    smooth solution.

    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.

    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Nfreqs,).

    flags : array_like, bool
        Boolean flags with the same shape as data.

    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.

        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.

    nmax: int or 2-tuple of ints
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu). If 2-tuple fit [-n0, n1].

    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.

    cn_guess : array_like, optional
        Initial guess for the series coefficients. If None, zeros will be used.
        A sensible choice of cn_guess can speed up the solver significantly.
        Default: None.

    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.

    operator : array_like, optional
        Fourier basis operator matrix. This is used to pass in a pre-computed
        matrix operator when calling from other functions, e.g. from
        delay_filter_leastsq. Operator must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be automatically
        calculated if no operator is specified.
    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.

    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes.

    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-nmax, +nmax].

    data_out : array_like
        In-painted data.
    """
    # Construct Fourier basis operator if not specified
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    if operator is None:
        F = fourier_operator(dsize=data.size, nmin = nmin, nmax=nmax, L=fundamental_period)
    else:
        F = operator
        cshape = nmax - nmin + 1
        if F.shape[0] != cshape:
            raise ValueError("Fourier basis operator has the wrong shape. "
                             "Must have shape (Nmodes, Nfreq).")
    # Turn flags into a mask
    w = np.logical_not(flags)

    # Define model and likelihood function
    def model(cn, F):
        return np.dot(cn, F)

    nmodes = nmax - nmin + 1

    # Initial guess for Fourier coefficients (real + imaginary blocks)
    cn_in = np.zeros(2 * nmodes)
    if cn_guess is not None:
        if cn_in.size != 2 * cn_guess.size:
            raise ValueError("cn_guess must be of size %s" % (cn_in.size / 2))
        cn_in[:cn_guess.shape[0]] = cn_guess.real
        cn_in[cn_guess.shape[0]:] = cn_guess.imag

    # Make sure sigma is the right size for matrix broadcasting
    if isinstance(sigma, np.ndarray):
        mat_sigma = np.tile(sigma, (nmodes, 1)).T
    else:
        mat_sigma = sigma

    # Run least-squares fit
    if use_linear:
        # Solve as linear system
        A = np.atleast_2d(w).T * F.T
        res = lsq_linear(A / mat_sigma ** 2., w * data / sigma ** 2.)
        cn_out = res.x
    else:
        # Use full non-linear leastsq fit
        def loglike(cn):
            """
            Simple log-likelihood, assuming Gaussian data. Calculates:
                logL = -0.5 [w*(data - model)]^2 / sigma^2.
            """
            # Need to do real and imaginary parts separately, otherwise
            # leastsq() fails
            _delta = w * (data - model(cn[:nmodes] + 1.j * cn[nmodes:], F))
            delta = np.concatenate((_delta.real / sigma, _delta.imag / sigma))
            return -0.5 * delta**2.

        # Do non-linear least-squares calculation
        cn, stat = leastsq(loglike, cn_in)
        cn_out = cn[:nmodes] + 1.j * cn[nmodes:]

    # Inject smooth best-fit model into masked areas
    bf_model = model(cn_out, F)
    data_out = data.copy()
    data_out[flags] = bf_model[flags]

    # Add noise to in-painted regions if requested
    if add_noise:
        noise = np.random.randn(np.sum(flags)) \
            + 1.j * np.random.randn(np.sum(flags))
        if isinstance(sigma, np.ndarray):
            data_out[flags] += sigma[flags] * noise
        else:
            data_out[flags] += sigma * noise

    # Return coefficients and best-fit model
    return bf_model, cn_out, data_out


def delay_filter_leastsq(data, flags, sigma, nmax, add_noise=False,
                         cn_guess=None, use_linear=True, operator=None, fundamental_period=None):
    """
    Fit a smooth model to each 1D slice of 2D complex-valued data with flags,
    using a linear least-squares solver. The model is a Fourier series up to a
    specified order. As well as calculating a best-fit model, this will also
    return a copy of the data with flagged regions filled in ('in-painted')
    with the smooth solution.

    Optionally, you can also add an uncorrelated noise realization on top of
    the smooth model in the flagged region.

    N.B. This is just a wrapper around delay_filter_leastsq_1d() but with some
    time-saving precomputations. It fits to each 1D slice of the data
    individually, and does not perform a global fit to the 2D data.

    Parameters
    ----------
    data : array_like, complex
        Complex visibility array as a function of frequency, with shape
        (Ntimes, Nfreqs).

    flags : array_like, bool
        Boolean flags with the same shape as data.

    sigma : float or array_like
        Noise standard deviation, in the same units as the data. If float,
        assumed to be homogeneous in frequency. If array_like, must have
        the same shape as the data.

        Note that the choice of sigma will have some bearing on how sensitive
        the fits are to small-scale variations.

    nmax: int
        Max. order of Fourier modes to fit. A model with complex Fourier modes
        between [-n, n] will be fitted to the data, where the Fourier basis
        functions are ~ exp(-i 2 pi n nu / (Delta nu).

    add_noise : bool, optional
        Whether to add an unconstrained noise realization to the in-painted areas.
        This uses sigma to set the noise standard deviation. Default: False.

    cn_guess : array_like, optional
        Initial guess for the series coefficients of the first row of the
        input data. If None, zeros will be used. Default: None.

    use_linear : bool, optional
        Whether to use a fast linear least-squares solver to fit the Fourier
        coefficients, or a slower generalized least-squares solver.
        Default: True.

    operator : array_like, optional
        Fourier basis operator matrix. Must have shape (Nmodes, Nfreq), where
        Nmodes = 2*nmax + 1. A complex Fourier basis will be used by default.
    fundamental_period : int, optional, default = None
        fundamental period of Fourier modes to fit too.
        if none, default to ndata.

    Returns
    -------
    model : array_like
        Best-fit model, composed of a sum of Fourier modes. Same shape as the
        data.

    model_coeffs : array_like
        Coefficients of Fourier modes, ordered from modes [-n, +n].

    data_out : array_like
        In-painted data.
    """
    if isinstance(nmax, tuple) or isinstance(nmax, list):
        nmin = nmax[0]
        nmax = nmax[1]
        assert isinstance(nmin, int) and isinstance(nmax, int), "Provide integers for nmax and nmin"
    elif isinstance(nmax, int):
        nmin = -nmax
    # Construct and cache Fourier basis operator (for speed)
    if operator is None:
        F = fourier_operator(dsize=data.shape[1], nmax=nmax, nmin=nmin, L=fundamental_period)
    else:
        # delay_filter_leastsq_1d will check for correct dimensions
        F = operator

    nmodes = nmax - nmin + 1
    # Array to store in-painted data
    inp_data = np.zeros(data.shape, dtype=np.complex)
    cn_array = np.zeros((data.shape[0], nmodes), dtype=np.complex)
    mdl_array = np.zeros(data.shape, dtype=np.complex)

    # Loop over array
    cn_out = None
    for i in range(data.shape[0]):
        bf_model, cn_out, data_out = delay_filter_leastsq_1d(
            data[i], flags[i], sigma=sigma, nmax=(nmin, nmax), add_noise=add_noise,
            use_linear=use_linear, cn_guess=cn_out, operator=F, fundamental_period=fundamental_period)
        inp_data[i, :] = data_out
        cn_array[i, :] = cn_out
        mdl_array[i, :] = bf_model

    return mdl_array, cn_array, inp_data


def _fit_basis_1d(x, y, w, filter_centers, filter_half_widths,
                basis_options, suppression_factors=None, hash_decimal=10,
                method='leastsq', basis='dft', cache=None):
    """
    A 1d linear-least-squares fitting function for computing models and residuals for fitting of the form
    y_model = A @ c
    where A is a design matrix encoding our choice for a basis functions
    and y_model is a fitted version of the data and c is a set of fitting coefficients determined by
    c = [A^T w A]^{-1} A^T w y
    where y is the original data and w is a diagonal matrix of weights for each channel in y.
    Currently supports fitting of dpss and dft modes.
    Parameters
    ----------
    x: array-like
        x-axis of data to fit.
    y: array-like
        y-axis of data to fit.
    w: array-like
        data weights.
    filter_centers': array-like
        list of floats specifying the centers of fourier windows with which to fit signals
    filter_half_widths': array-like
        list of floats specifying the half-widths of fourier windows to model.
    suprression_factors: array-like, optional
        list of floats for each basis function denoting the fraction of
        of each basis element that should be present in the fitted model
        If none provided, model will include 100% of each mode.
        It is sometimes useful, for renormalization reversability
        to only include 1-\epsilon where \epsilon is a small number of
        each mode in the model.
    hash_decimal: number of decimals to round to for floating point keys.
    basis_options: dictionary
        basis specific options for fitting. The two bases currently supported are dft and dpss whose options
        are as follows:
            * 'dft':
               *'fundamental_period': float or 2-tuple
                The fundamental_period of dft modes to fit. This is the
                Fourier resoltion of fitted fourier modes equal to
                1/FP where FP is the fundamental period. For a standard
                delay DFT FP = B where B is the visibility bandwidth
                FP also sets the number of
                modes fit within each window in 'filter_half_widths' will
                equal fw / fundamental_period where fw is the filter width.
                if filter2d, must provide a 2-tuple with fundamental_period
                of each dimension.
            * 'dpss':
                The basis_options must include one and only one of the four options
                for specifying how to terminate the dpss series in each filter window.
                *'eigenval_cutoff': array-like
                    list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                *'nterms': array-like
                    list of integers specifying the order of the dpss sequence to use in each
                    filter window.
                *'edge_supression': array-like
                    specifies the degree of supression that must occur to tones at the filter edges
                    to calculate the number of DPSS terms to fit in each sub-window.
                *'avg_suppression': list of floats, optional
                    specifies the average degree of suppression of tones inside of the filter edges
                    to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                    the suppression of a since vector with equal contributions from all tones inside of the
                    filter width instead of a single tone.
    method: string
        specifies the fitting method to use. We currently support.
            *'leastsq' to perform iterative leastsquares fit to derive model.
                using scipy.optimize.leastsq
            *'matrix' derive model by directly calculate the fitting matrix
                [A^T W A]^{-1} A^T W and applying it to the y vector.


    Returns:
        model: array-like
            Ndata array of complex floats equal to interpolated model
        resid: array-like
            Ndata array of complex floats equal to y - model
        info:
            dictionary containing fitting arguments for reference.
            if 'matrix' method is used. Fields are
                * 'method' : method used to derive fits.
                * 'basis' : basis that the fits are in
                * 'filter_centers' : filtering centers argument
                * 'filter_half_widths' : filter_half_widths argument
                * 'suppression_factors' : suppression_factors argument
                * 'basis_options' : basis specific options dictionary
                                    see dpss_operator and dft_operator.
                * 'amat' : A matrix used for fitting.
                * 'fitting_matrix' : matrix used for fitting (A [ATA]^-1 AT)
                  if the method == 'matrix'.

    """
    if cache is None:
        cache = {}
    info = copy.deepcopy(basis_options)
    if basis.lower() == 'dft':
        amat = dft_operator(x, filter_centers=filter_centers,
                            filter_half_widths=filter_half_widths,
                            cache=cache, **basis_options)
    elif basis.lower() == 'dpss':
        amat, nterms = dpss_operator(x, filter_centers=filter_centers,
                                     filter_half_widths=filter_half_widths,
                                     cache=cache, **basis_options)
        info['nterms'] = nterms
    else:
        raise ValueError("Specify a fitting basis in supported bases: ['dft', 'dpss']")
    if suppression_factors is None:
        suppression_vector = np.ones(amat.shape[1])
    else:
        if basis.lower() == 'dft':
            suppression_vector =  np.hstack([1-sf * np.ones(2*int(np.ceil(fw * basis_options['fundamental_period'])))\
                                             for sf,fw in zip(suppression_factors, filter_half_widths)])
        elif basis.lower() == 'dpss':
            suppression_vector = np.hstack([1-sf * np.ones(nterm) for sf, nterm in zip(suppression_factors, nterms)])
    info['method'] = method
    info['basis'] = basis
    info['filter_centers'] = filter_centers
    info['filter_half_widths'] = filter_half_widths
    info['suppression_factors'] = suppression_factors
    info['basis_options'] = basis_options
    info['amat'] = amat
    info['skipped'] = False
    wmat = np.diag(w)
    if method == 'leastsq':
        a = np.atleast_2d(w).T * amat
        try:
            res = lsq_linear(a, w * y)
            cn_out = res.x
        # np.linalg.LinAlgError catches "SVD did not converge."
        # which can happen if the solution is under-constrained.
        # also handle nans and infs in the data here too.
        except (np.linalg.LinAlgError, ValueError, TypeError) as err:
            warn(f"{err} -- recording skipped integration in info and setting to zero.")
            cn_out = 0.0
            info['skipped'] = True
    elif method == 'matrix':
        fm_key = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                      filter_factors=suppression_vector, x=x, w=w, hash_decimal=hash_decimal,
                                      label='fitting matrix', basis=basis)
        if basis.lower() == 'dft':
            fm_key = fm_key + (basis_options['fundamental_period'], )
        elif basis.lower() == 'dpss':
            fm_key = fm_key + tuple(nterms)
        fmat = fit_solution_matrix(wmat, amat, cache=cache, fit_mat_key=fm_key)
        info['fitting_matrix'] = fmat
        cn_out = fmat @ y
    else:
        raise ValueError("Provided 'method', '%s', is not in ['leastsq', 'matrix']."%(method))
    model = amat @ (suppression_vector * cn_out)
    resid = (y - model) * (~np.isclose(w, 0, atol=1e-10)).astype(float) #suppress flagged residuals (such as RFI)
    return model, resid, info

def _clean_filter(x, data, wgts, filter_centers, filter_half_widths,
                  clean2d=False, tol=1e-9, window='none', skip_wgt=0.1,
                  maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                  edgecut_low=0, edgecut_hi=0, add_clean_residual=False,
                  zero_residual_flags=True):
    '''
    core cleaning functionality
    Input sanitation not implemented. Should be called through
    fourier_filter and the higher level functions that call fourier_filter.

    Parameters
    ----------
    x : array-like (or 2-tuple/list of arrays for filter2d)
        x-values of data to be cleaned. Each x-axis must be equally spaced.
    data : array-like, complex, 1d or 2d numpy array of data to be filtered.
    wgts : array-like, float, 1d or 2d numpy array of wgts for data.
    filter_centers : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     centers of filtering regions in units of 1 / x-units
    filter_half_widths : list of floats (1d clean) 2-list of lists of floats (2d clean)
                     half-widths of filtering regions in units of 1 / x-units
    clean2d : bool, optional, specify if 2dclean is to be performed.
              if False, just clean axis -1.
    tol : float, tolerance parameter for clean.
    window : str, apodization to perform on data before cleaning.
    skip_wgt : float, If less then skip_wgt fraction of data is flagged, skip the clean.
    maxiter : int, maximum number of clean iterations.
    gain : float, fraction of detected peak to subtract on each clean iteration.
    filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
        If 'rect', a 2D rectangular filter is constructed in fourier space (default).
        If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
        slice along 0 delay and fringe-rate is kept.
    edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_low for first and second FFT axis.
    edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
        such that the windowing function smoothly approaches zero. For 2D cleaning, can
        be fed as a tuple specifying edgecut_hi for first and second FFT axis.
    add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
        in fourier space to the CLEAN model. Note that the residual actually returned is
        not the CLEAN residual, but the residual in input data space.
    zero_residual_flags : bool, optional.
        If true, set flagged channels in the residual equal to zero.
        Default is True.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    import aipy
    if not clean2d:
        #pad = [0, pad]
        _x = [np.zeros(data.shape[0]), np.fft.fftfreq(len(x), x[1]-x[0])]
        x = [np.zeros(data.shape[0]), x]
        edgecut_hi = [ 0, edgecut_hi ]
        edgecut_low = [ 0, edgecut_low ]
        filter_centers = [[0.], copy.deepcopy(filter_centers)]
        filter_half_widths = [[np.inf], copy.deepcopy(filter_half_widths)]
        window = ['none', window]
    else:
        if not np.all(np.isclose(np.diff(x[1]), np.mean(np.diff(x[1])))):
            raise ValueError("Data must be equally spaced for CLEAN mode!")
        _x = [np.fft.fftfreq(len(x[m]), x[m][1]-x[m][0]) for m in range(2)]
        #window_opt = window
    for m in range(2):
        if not np.all(np.isclose(np.diff(x[m]), np.mean(np.diff(x[m])))):
            raise ValueError("Data must be equally spaced for CLEAN mode!")
    window = [gen_window(window[m], data.shape[m], alpha=alpha,
                       edgecut_low=edgecut_low[m], edgecut_hi=edgecut_hi[m]) for m in range(2)]
    window[0] = np.atleast_2d(window[0]).T
    area_vecs = [ np.zeros(len(_x[m])) for m in range(2) ]
    #set area equal to one inside of filtering regions
    info = {}
    info['filter_params'] = {'axis_0':{}, 'axis_1':{}}
    info['clean_status'] = {'axis_0':{}, 'axis_1':{}}
    info['status'] = {'axis_0':{}, 'axis_1':{}}
    if filt2d_mode == 'rect' or not clean2d:
        for m in range(2):
            for fc, fw in zip(filter_centers[m], filter_half_widths[m]):
                area_vecs[m] = _get_filter_area(x[m], fc, fw)
        area = np.outer(area_vecs[0], area_vecs[1])
    elif filt2d_mode == 'plus' and clean2d:
        area = np.zeros(data.shape)
        #construct and add a 'plus' for each filtering window pair in each dimension.
        for fc0, fw0 in zip(filter_centers[0], filter_half_widths[0]):
            for fc1, fw1 in zip(filter_centers[1], filter_half_widths[1]):
                area_temp = np.zeros(area.shape)
                if fc0 >= _x[0].min() and fc0 <= _x[0].max():
                    #generate area vector centered at zero
                    av = _get_filter_area(x[1], fc1, fw1)
                    area_temp[np.argmin(np.abs(_x[0]-fc0)), :] = av
                if fc1 >= _x[1].min() and fc1 <= _x[1].max():
                    #generate area vector centered at zero
                    av = _get_filter_area(x[0], fc0, fw0)
                    area_temp[:, np.argmin(np.abs(_x[1]-fc1))] = av
                area += area_temp
        area = (area>0.).astype(int)
    else:
        raise ValueError("%s is not a valid filt2d_mode! choose from ['rect', 'plus']"%(filt2d_mode))
    if clean2d:
        _wgts = np.fft.ifft2(window[0] * wgts * window[1])
        _data = np.fft.ifft2(window[0] * data * wgts * window[1])
    else:
        _wgts = np.fft.ifft(window[0] * wgts * window[1], axis=1)
        _data = np.fft.ifft(window[0] * wgts * data * window[1], axis=1)
    _d_cl = np.zeros_like(_data)
    _d_res = np.zeros_like(_data)
    if not clean2d:
        for i, _d, _w, _a in zip(np.arange(_data.shape[0]).astype(int), _data, _wgts, area):
            # we skip steps that might trigger infinite CLEAN loops or divergent behavior.
            # if the weights sum up to a value close to zero (most of the data is flagged)
            # or if the data itself is close to zero.
            if _w[0] < skip_wgt or np.all(np.isclose(_d, 0.)):
                _d_cl[i] = 0.
                _d_res[i] = _d
                info['status']['axis_1'][i] = 'skipped'
            else:
                _d_cl[i], _info = aipy.deconv.clean(_d, _w, area=_a, tol=tol, stop_if_div=False,
                                                maxiter=maxiter, gain=gain)
                _d_res[i] = _info['res']
                _info['skipped'] = False
                del(_info['res'])
                info['clean_status']['axis_1'][i] = _info
                info['status']['axis_1'][i] = 'success'
    elif clean2d:
            # we skip 2d cleans if all the data is close to zero (which can cause an infinite clean loop)
            # or the weights are all equal to zero which can also lead to a clean loop.
            # the maximum of _wgts should be the average value of all cells in 2d wgts.
            # since it is the 2d fft of wgts.
            if not np.all(np.isclose(_data, 0.)) and np.abs(_wgts).max() > skip_wgt:
                _d_cl, _info = aipy.deconv.clean(_data, _wgts, area=area, tol=tol, stop_if_div=False,
                                                maxiter=maxiter, gain=gain)
                _d_res = _info['res']
                del(_info['res'])
                info['clean_status']['axis_1'] = _info
                info['clean_status']['axis_0'] = info['clean_status']['axis_1']
                info['status']['axis_1'] = {i:'success' for i in range(_data.shape[0])}
                info['status']['axis_0'] = {i:'success' for i in range(_data.shape[1])}
            else:
                info['clean_status']['axis_0'] = {'skipped':True}
                info['clean_status']['axis_1'] = {'skipped':True}
                info['status']['axis_1'] = {i:'skipped' for i in range(_data.shape[0])}
                info['status']['axis_0'] = {i:'skipped' for i in range(_data.shape[1])}
                _d_cl = np.zeros_like(_data)
                _d_res = np.zeros_like(_d_cl)
    if add_clean_residual:
        _d_cl = _d_cl + _d_res * area
    if clean2d:
        model = np.fft.fft2(_d_cl)
    else:
        model = np.fft.fft(_d_cl, axis=1)
    #transpose back if filtering the 0th dimension.
    residual = (data - model)
    if zero_residual_flags:
        windmat = np.outer(window[0], window[1])
        residual *= (~np.isclose(wgts * windmat, 0.0, atol=1e-10)).astype(float)
    return model, residual, info




def _fit_basis_2d(x, data, wgts, filter_centers, filter_half_widths,
                basis_options, suppression_factors=None,
                method='leastsq', basis='dft', cache=None,
                filter_dims = 1, skip_wgt=0.1, max_contiguous_edge_flags=5,
                zero_residual_flags=True):
    """
    A 1d linear-least-squares fitting function for computing models and residuals for fitting of the form
    y_model = A @ c
    where A is a design matrix encoding our choice for a basis functions
    and y_model

    Parameters
    ----------
    x: array-like or 2-tuple/2-list
        x-axis of data to fit.
        if more then one filter_dim, must provide 2-tuple or 2-list with x
    data: array-like
        data to fit, should be an Ntimes x Nfreqs array.
    wgts: array-like
        data weights.
    filter_centers': array-like
        list of floats specifying the centers of fourier windows with which to fit signals
    filter_half_widths': array-like
        list of floats specifying the half-widths of fourier windows to model.
    suppression_factors: array-like, optional
        list of floats for each basis function denoting the fraction of
        of each basis element that should be present in the fitted model
        If none provided, model will include 100% of each mode.
        It is sometimes useful, for renormalization reversability
        to only include 1-\epsilon where \epsilon is a small number of
        each mode in the model.
    basis_options: dictionary
        basis specific options for fitting. The two bases currently supported are dft and dpss whose options
        are as follows:
            * 'dft':
              *'fundamental_period': float or 2-list/tuple
                The fundamental_period of dft modes to fit. This is the
                Fourier resoltion of fitted fourier modes equal to
                1/FP where FP is the fundamental period. For a standard
                delay DFT FP = B where B is the visibility bandwidth
                FP also sets the number of
                modes fit within each window in 'filter_half_widths' will
                equal fw / fundamental_period where fw is the filter width.
                if filter2d, must provide a 2-tuple with fundamental_period
                of each dimension.
            * 'dpss':
                The basis_options must include one and only one of the four options
                for specifying how to terminate the dpss series in each filter window.
                *'eigenval_cutoff': array-like
                    list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                    if 2d fit, should be a 2-list with each element giving list
                    of eigenval cutoffs for each dimension.
                *'nterms': array-like
                    list of integers specifying the order of the dpss sequence to use in each
                    filter window. if 2d fit, should be a 2-list of lists of nterms for each delay
                    window in each dimension.
                *'edge_supression': array-like
                    specifies the degree of supression that must occur to tones at the filter edges
                    to calculate the number of DPSS terms to fit in each sub-window.
                    if 2d fit, should be a 2-list of lists of edge_suppression thresholds in each dim
                *'avg_suppression': list of floats, optional
                    specifies the average degree of suppression of tones inside of the filter edges
                    to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                    the suppression of a since vector with equal contributions from all tones inside of the
                    filter width instead of a single tone.
                    if 2d fit, should be a list of lists of avg_suppression thressholds for each.
    method: string
        specifies the fitting method to use. We currently support.
            *'leastsq' to perform iterative leastsquares fit to derive model.
                using scipy.optimize.leastsq
            *'matrix' derive model by directly calculate the fitting matrix
                [A^T W A]^{-1} A^T W and applying it to the y vector.
    filter_dim, int optional
        specify dimension to filter. default 1,
        and if 2d filter, will use both dimensions.
    skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
        time. Only works properly when all weights are all between 0 and 1.
    max_contiguous_edge_flags : int, optional
        if the number of contiguous samples at the edge is greater then this
        at either side, skip .
    zero_residual_flags : bool, optional.
        If true, set flagged channels in the residual equal to zero.
        Default is True.
    Returns
    -------
        model: array-like
            Ndata array of complex floats equal to interpolated model
        resid: array-like
            Ndata array of complex floats equal to y - model
    info: dictionary with filtering parameters and a list of skipped_times and skipped_channels
          has the following fields
         * 'status': dict holding two sub-dicts status of filtering on each time/frequency step.
                   - 'axis_0'/'axis_1': dict holding the status of time filtering for each time/freq step. Keys are integer index
                               of each step and values are a string that is either 'success' or 'skipped'.
         * 'filter_params': dict holding the filtering parameters for each axis with the following sub-dicts.
                   - 'axis_0'/'axis_1': dict holding filtering parameters for filtering over each respective axis.
                               - 'filter_centers': centers of filtering windows.
                               - 'filter_half_widths': half-widths of filtering regions for each axis.
                               - 'suppression_factors': amount of suppression for each filtering region.
                               - 'x': vector of x-values used to generate the filter.
                               - 'basis': (if using dpss/dft) gives the filtering basis.
                               - 'basis_options': the basis options used for dpss/dft mode. See dft_operator and dpss_operator for
                                                  more details.
    """
    if isinstance(filter_dims, (int, np.integer)):
        filter_dims = [filter_dims]
    if cache is None:
        cache={}
    info = {'status':{'axis_0':{}, 'axis_1':{}}}
    residual = np.zeros_like(data)
    filter2d = (0 in filter_dims and 1 in filter_dims)
    filter_dims = sorted(filter_dims)[::-1]
    #this will only happen if filter_dims is only zero!
    if filter_dims[0] == 0:
        data = data.T
        wgts = wgts.T
    if not filter2d:
        x = [np.zeros_like(x), x]
        filter_centers = [[], copy.deepcopy(filter_centers)]
        filter_half_widths = [[], copy.deepcopy(filter_half_widths)]
        suppression_factors = [[], copy.deepcopy(suppression_factors)]
        basis_options=[{}, basis_options]
    else:
        for k in basis_options:
            if not isinstance(basis_options[k], (tuple,list)) or not len(basis_options[k]) == 2:
                raise ValueError("basis_options values must be 2-tuple or 2-list for 2d filtering.")
        basis_options = [{k:basis_options[k][0] for k in basis_options}, {k:basis_options[k][1] for k in basis_options}]
    #filter -1 dimension
    model = np.zeros_like(data)
    for i, _y, _w, in zip(range(data.shape[0]), data, wgts):
        if np.count_nonzero(_w)/len(_w) >= skip_wgt and np.count_nonzero(_w[:max_contiguous_edge_flags]) > 0 \
                                                        and np.count_nonzero(_w[-max_contiguous_edge_flags:]) >0:
            model[i], _, info_t = _fit_basis_1d(x=x[1], y=_y, w=_w, filter_centers=filter_centers[1],
                                            filter_half_widths=filter_half_widths[1],
                                            suppression_factors=suppression_factors[1],
                                            basis_options=basis_options[1], method=method,
                                            basis=basis, cache=cache)
            if info_t['skipped']:
                info['status']['axis_1'][i] = 'skipped'
            else:
                info['status']['axis_1'][i] = 'success'
        else:
            info['status']['axis_1'][i] = 'skipped'
    #and if filter2d, filter the 0 dimension. Note that we feed in the 'model'
    #set wgts for time filtering to happen on skipped rows
    info['filter_params'] = {'axis_0':{}, 'axis_1':{}}
    if np.any([info['status']['axis_1'][i] == 'success' for i in info['status']['axis_1']]):
        info['filter_params']['axis_1']['method'] = info_t['method']
        info['filter_params']['axis_1']['basis'] = info_t['basis']
        info['filter_params']['axis_1']['filter_centers'] = info_t['filter_centers']
        info['filter_params']['axis_1']['filter_half_widths'] = info_t['filter_half_widths']
        info['filter_params']['axis_1']['suppression_factors'] = info_t['suppression_factors']
        info['filter_params']['axis_1']['basis_options'] = info_t['basis_options']
        info['filter_params']['axis_1']['mode'] = info_t['basis'] + '_' + method
    if filter2d:
        wgts_time = np.ones_like(wgts)
        for i in range(data.shape[0]):
            if info['status']['axis_1'][i] == 'skipped':
                wgts_time[i] = 0.
        for i, _y, _w, in zip(range(model.shape[1]), model.T, wgts_time.T):
            if np.count_nonzero(_w)/len(_w) >= skip_wgt and np.count_nonzero(_w[:max_contiguous_edge_flags]) > 0 \
               and np.count_nonzero(_w[-max_contiguous_edge_flags:]) >0:
                model.T[i], _, info_t = _fit_basis_1d(x=x[0], y=_y, w=_w, filter_centers=filter_centers[0],
                                                                 filter_half_widths=filter_half_widths[0],
                                                                 suppression_factors=suppression_factors[0],
                                                                 basis_options=basis_options[0], method=method,
                                                                 basis=basis, cache=cache)
                if info_t['skipped']:
                    info['status']['axis_0'][i] = 'skipped'
                else:
                    info['status']['axis_0'][i] = 'success'
            else:
                info['status']['axis_0'][i] = 'skipped'
        if np.any([info['status']['axis_0'][i] == 'success' for i in info['status']['axis_0']]):
            info['filter_params']['axis_0']['method'] = info_t['method']
            info['filter_params']['axis_0']['basis'] = info_t['basis']
            info['filter_params']['axis_0']['filter_centers'] = info_t['filter_centers']
            info['filter_params']['axis_0']['filter_half_widths'] = info_t['filter_half_widths']
            info['filter_params']['axis_0']['suppression_factors'] = info_t['suppression_factors']
            info['filter_params']['axis_0']['basis_options'] = info_t['basis_options']

    residual = (data - model) * (np.abs(wgts) > 0).astype(float)
    #this will only happen if filter_dims is only zero!
    if filter_dims[0] == 0:
        data = data.T
        wgts = wgts.T
        model = model.T
        residual = residual.T
        for k in info:
            info[k]['axis_0'] = copy.deepcopy(info[k]['axis_1'])
            info[k]['axis_1'] = {}
    if zero_residual_flags:
        residual = residual * (~np.isclose(wgts, 0., atol=1e-10)).astype(float) # set residual to zero in flags.
    return model, residual, info


def fit_solution_matrix(weights, design_matrix, cache=None, hash_decimal=10, fit_mat_key=None):
    """
    Calculate the linear least squares solution matrix
    from a design matrix, A and a weights matrix W
    S = [A^T W A]^{-1} A^T W

    Parameters
    ----------
    weights: array-like
        ndata x ndata matrix of data weights
    design_matrx: array-like
        ndata x n_fit_params matrix transforming fit_parameters to data
    cache: optional dictionary
        optional dictionary storing pre-computed fitting matrix.
    hash_decimal: int optional
        the number of decimals to use in hash for caching. default is 10
    fit_mat_key: optional hashable variable
        optional key. If none is used, hash fit matrix against design and
        weighting matrix.

    Returns
    -----------
        array-like
        n_fit_params x n_fit_params matrix
        S = [A^T W A]^{-1} A ^T W
    """
    if cache is None:
        cache = {}
    ndata = weights.shape[0]
    if not weights.shape[0] == weights.shape[1]:
        raise ValueError("weights must be a square matrix")
    if not design_matrix.shape[0] == ndata:
        raise ValueError("weights matrix incompatible with design_matrix!")
    if fit_mat_key is None:
            opkey = ('fitting_matrix',) + tuple(np.round(weights.flatten(), hash_decimal))\
                    +tuple(np.round(design_matrix.flatten(), hash_decimal))
    else:
        opkey = fit_mat_key

    if not opkey in cache:
        #check condition number
        cmat = np.conj(design_matrix.T) @ weights @ design_matrix
        #should there be a conjugation!?!
        if np.linalg.cond(cmat)>=1e9:
            warn('Warning!!!!: Poorly conditioned matrix! Your linear inpainting IS WRONG!')
            cache[opkey] = np.linalg.pinv(cmat) @ np.conj(design_matrix.T) @ weights
        else:
            try:
                cache[opkey] = np.linalg.inv(cmat) @ np.conj(design_matrix.T) @ weights
            except np.linalg.LinAlgError as error:
                print(error)
                cache[opkey] = None
    return cache[opkey]


def dpss_operator(x, filter_centers, filter_half_widths, cache=None, eigenval_cutoff=None,
        edge_suppression=None, nterms=None, avg_suppression=None, xc=None, hash_decimal=10,
        xtol=1e-3):
    """
    Calculates DPSS operator with multiple delay windows to fit data. Frequencies
    must be equally spaced (unlike Fourier operator). Users can specify how the
    DPSS series fits are cutoff in each delay-filtering window with one (and only one)
    of three conditions: eigenvalues in sinc matrix fall below a thresshold (eigenval_cutoff),
    user specified number of DPSS terms (nterms), xor the suppression of fourier
    tones at the filter edge by a user specified amount (edge_supression).

    Parameters
    ----------
    x: array-like
        x values to evaluate operator at
    filter_centers: array-like
        list of floats of centers of delay filter windows in nanosec
    filter_half_widths: array-like
        list of floats of half-widths of delay filter windows in nanosec
    cache: dictionary, optional
        dictionary for storing operator matrices with keys
        tuple(x) + tuple(filter_centers) + tuple(filter_half_widths)\
         + (series_cutoff_name,) = tuple(series_cutoff_values)
    eigenval_cutoff: list of floats, optional
        list of sinc matrix eigenvalue cutoffs to use for included dpss modes.
    nterms: list of integers, optional
        integer specifying number of dpss terms to include in each delay fitting block.
    edge_suppression: list of floats, optional
        specifies the degree of supression that must occur to tones at the filter edges to
        calculate the number of DPSS terms to fit in each sub-window.
    avg_suppression: list of floats, optional
        specifies the average degree of suppression of tones inside of the filter edges
        to calculate the number of DPSS terms. Similar to edge_suppression but instead
        checks the suppression of a sinc vector with equal contributions from
        all tones inside of the filter width instead of a single tone.
    xc: float optional
    hash_decimal: number of decimals to round for floating point dict keys.
    xtol: fraction of average diff that the diff between all x-values must be within
          the average diff to be considered
          equally spaced. Default is 1e-3

    Returns
    ----------
    2-tuple
    First element:
        Design matrix for DPSS fitting.   Ndata x (Nfilter_window * nterm)
        transforming from DPSS modes to data.
    Second element:
        list of integers with number of terms for each fourier window specified by filter_centers
        and filter_half_widths
    """
    if cache is None:
        cache = {}
    #conditions for halting.
    crit_labels = ['eigenval_cutoff', 'nterms', 'edge_suppression', 'avg_suppression']
    crit_list = [eigenval_cutoff, nterms, edge_suppression, avg_suppression]
    crit_provided = np.asarray([not crit is None for crit in crit_list]).astype(bool)
    #only allow the user to specify a single condition for cutting off DPSS modes to fit.
    crit_provided_name = [ label for m,label in enumerate(crit_labels) if crit_provided[m] ]
    crit_provided_value = [ crit for m,crit in enumerate(crit_list) if crit_provided[m] ]
    if np.count_nonzero(crit_provided) != 1:
        raise ValueError('Must only provide a single series cutoff condition. %d were provided: %s '%(np.count_nonzero(crit_provided),
                                                                                                 str(crit_provided_name)))
    opkey = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                 filter_factors=[0.], crit_name=crit_provided_name[0], x=x,
                                 w=None, hash_decimal=hash_decimal,
                                 label='dpss_operator', crit_val=tuple(crit_provided_value[0]))
    if not opkey in cache:
        # try placing x on a uniform grid.
        # x is a version of x with the in-between grid values filled in and inserted is a boolean vector
        # set to True wherever a value for x was inserted and False otherwise.
        x, _, _, inserted = place_data_on_uniform_grid(x, np.zeros(len(x)), np.ones(len(x)))
        # if this is not successful, then throw a value error..
        if not np.allclose(np.diff(x), np.median(np.diff(x)), rtol=0., atol=np.abs(xtol * np.median(np.diff(x)))):
            #for now, don't support DPSS iterpolation unless x is equally spaced.
            #In principal, I should be able to compute off-grid DPSS points using
            #the fourier integral of the DPSWF
            raise ValueError('x values must be equally spaced for DPSS operator!')
        nf = len(x)
        df = np.abs(x[1]-x[0])
        xg, yg = np.meshgrid(x,x)
        if xc is None:
            xc = x[nf//2]
        #determine cutoffs
        if nterms is None:
            nterms = []
            for fn,fw in enumerate(filter_half_widths):
                dpss_vectors = windows.dpss(nf, nf * df * fw, nf)
                if not eigenval_cutoff is None:
                    smat = np.sinc(2 * fw * (xg-yg)) * 2 * df * fw
                    eigvals = np.sum((smat @ dpss_vectors.T) * dpss_vectors.T, axis=0)
                    nterms.append(np.max(np.where(eigvals>=eigenval_cutoff[fn])))
                if not edge_suppression is None:
                    z0=fw * df
                    edge_tone=np.exp(-2j*np.pi*np.arange(nf)*z0)
                    fit_components = dpss_vectors * (dpss_vectors @ edge_tone)
                    #this is a vector of RMS residuals of a tone at the edge of the delay window being fitted between 0 to nf DPSS components.
                    rms_residuals = np.asarray([ np.sqrt(np.mean(np.abs(edge_tone - np.sum(fit_components[:k],axis=0))**2.)) for k in range(nf)])
                    nterms.append(np.max(np.where(rms_residuals>=edge_suppression[fn])))
                if not avg_suppression is None:
                    sinc_vector=np.sinc(2 * fw * df * (np.arange(nf)-nf/2.))
                    sinc_vector = sinc_vector / np.sqrt(np.mean(sinc_vector**2.))
                    fit_components = dpss_vectors * (dpss_vectors @ sinc_vector)
                    #this is a vector of RMS residuals of vector with equal contributions from all tones within -fw and fw.
                    rms_residuals = np.asarray([ np.sqrt(np.mean(np.abs(sinc_vector - np.sum(fit_components[:k],axis=0))**2.)) for k in range(nf)])
                    nterms.append(np.max(np.where(rms_residuals>=avg_suppression[fn])))
        #next, construct A matrix.
        amat = []
        for fc, fw, nt in zip(filter_centers,filter_half_widths, nterms):
            amat.append(np.exp(2j * np.pi * (yg[:,:nt]-xc) * fc ) * windows.dpss(nf, nf * df * fw, nt).T )
        if len(amat) > 1:
            amat = np.hstack(amat)
        else:
            amat = amat[0]
        # we used the regularly spaced inserted grid to generate our fitting basis vectors
        # but we dont need them for the actual fit.
        # so here we keep only the non-inserted rows of the design matrix.
        amat = amat[~inserted, :]
        cache[opkey] = (amat, nterms)
    return cache[opkey]


def dft_operator(x, filter_centers, filter_half_widths,
                cache=None, fundamental_period=None, xc=None, hash_decimal=10):
    """
    Discrete Fourier operator with multiple flexible delay windows to fit data, potentially with arbitrary
    user provided frequencies.

    A_{nu tau} = e^{- 2 * pi * i * nu * tau / B}

    for a set of taus contained within delay regions centered at filter_centers
    and with half widths of filter_half_widths separated by 1/B where B
    is provided by fundamental_period.

    Parameters
    ----------
    x: array-like floats.
        x values to evaluate operator at
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    cache: dictionary, optional dictionary storing operator matrices with keys
    (x) + (filter_centers) + (filter_half_widths) + \
    hash_decimal: int, optional number of decimals to use for floating point keys.

    Returns
    --------
    Ndata x (Nfilter_window * nterm) design matrix transforming DFT coefficients
    to data.

    """
    if cache is None:
        cache = {}
    #if no fundamental fourier period is provided, set fundamental period equal to measurement
    #bandwidth.
    if fundamental_period is None:
        fundamental_period = np.median(np.diff(x)) * len(x)
    if xc is None:
        xc = x[int(np.round(len(x)/2))]
    if isinstance(filter_centers, float):
        filter_centers = [filter_centers]
    if isinstance(filter_half_widths, float):
        filter_half_widths = [filter_half_widths]

    #each column is a fixed delay
    opkey = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                 filter_factors=[0.], x=x, w=None, hash_decimal=hash_decimal,
                                 label='dft_operator', fperiod=fundamental_period)
    if not opkey in cache:
        amat = []
        for fc, fw in zip(filter_centers,filter_half_widths):
            bs = np.ceil(fw * fundamental_period)
            dlys = fc + np.arange(-bs, bs) / fundamental_period
            xg, dg = np.meshgrid(x-xc, dlys, indexing='ij')
            fblock = np.exp(2j * np.pi * dg * xg)
            amat.append(fblock)
        cache[opkey] = np.hstack(amat)
    return cache[opkey]



def delay_interpolation_matrix(nchan, ndelay, wgts, fundamental_period=None, cache=None, window='none'):
    """
    Compute a foreground interpolation matrix.

    Computes a foreground interpolation matrix that, when applied to data,
    interpolates over flagged channels with delays between
    -ndelay / fundamental_period, ndelay / fundamental_period

    The computed Matrix is equal to F = A @ [ A^T @ W @ A]^{-1} @ A^T W
    where A is an nchan \times 2ndelay  design matrix
    y = A \tilde{y}
    y is the frequency representation of data and \tilde{y} is
    a 2xndelay vector holding the data's fourier coefficients. W is a diagonal
    matrix of frequency-data weights. The net effect of F, when applied to flagged
    data, is to solve for the fourier coefficients fitting unflagged channels
    ([ A^T @ W @ A]^{-1} @ A^T W solves the linear least squares problem) and then return
    the unflagged Fourier transform by apply A @ to the fitted coefficients, resulting
    in data that is linearly interpolated.

    !!! THIS FUNCTION WILL BE DEPRECATED BY fit_solution_matrix !!!

    Parameters
    ----------
    nchan: int
        Number of frequency channels to interpolate over.
    ndelay: int
        number of delays to use in interpolation.
    wgts: float array
        wgts to be applied to each frequency channel.
        must have length equal to nchan.
        in addition, wgts should have more nonezero values then there are
        degrees of freedom (delay modes) to solve for.
    fundamental_period: float, optional
        fundamental period of Fourier modes to fit too.
        this sets the resolution in Fourier space. A standard DFT has a resolution
        of 1/N_{FP} = 1/N between fourier modes so that the DFT operator is
        D_{mn} = e^{-2 \pi i m n / N_{FP}}. fg_deconv_fundamental_period
        is N_{FP}.
    cache: dict, optional
        optional cache holding pre-computed matrices
    window: string, optional
        use a window to fit.
    Returns
    ----------
    (nchan, nchan) numpy array
        that can be used to interpolate over channel gaps.
    """
    if cache is None:
        cache = {}
    if not len(wgts) == nchan:
        raise ValueError("nchan must equal length of wgts")
    if fundamental_period is None: #recommend 2 x nchan or nchan.
        fundamental_period = 2*nchan #this tends to give well conditioned matrices.
    if not np.sum((np.abs(wgts) > 0.).astype(float)) >= 2*ndelay:
        raise ValueError("number of unflagged channels must be greater then or equal to number of delays")
    matkey = (nchan, ndelay, fundamental_period) + tuple(wgts)
    amat = dft_operator(x=np.arange(nchan)-nchan/2., filter_centers=[0.], filter_half_widths=[ndelay/fundamental_period],
                                          cache=cache, fundamental_period=fundamental_period)
    wmat = np.diag(wgts * gen_window(window, nchan)).astype(complex)
    fs = fit_solution_matrix(wmat, amat)
    if fs is not None:
        return amat @ fs
    else:
        return np.nan * np.ones((nchan, nchan))


def dayenu_mat_inv(x, filter_centers, filter_half_widths,
                            filter_factors, cache=None, wrap=False, wrap_interval=1,
                            nwraps=1000, no_regularization=False, hash_decimal=10):
    """
    Computes the inverse of sinc weights for a baseline.
    This form of weighting is diagonal in delay-space and down-weights tophat regions.

    Parameters
    ----------
    x: array like
        array-like list of arbitrary frequencies. If this is supplied, evaluate sinc_downweight_mat at these frequencies
        instead of linear array of nchan.
    filter_centers: float or list
        float or list of floats of centers of delay filter windows in nanosec
    filter_half_widths: float or list
        float or list of floats of half-widths of delay filter windows in nanosec
    filter_factors: float or list
        float or list of floats of filtering factors.
    cache: dictionary, optional dictionary storing filter matrices with keys
    hash_decimal int, number of decimals to consider when hashing x
    tuple(x) + (filter_centers) + (filter_half_widths) + \
    (filter_factors)



    !!!-------------
    WARNING: The following parameters are intended for theoretical
    studies of how inverse sinc-weighting functions
    but should not be changed from defaults in practical data analysis!
    !!!------------
        wrap: bool, If true, add a wrap around, equivalent to situation
              where we want sinc weights to be the IDFT of a diagonal matrix
        wrap_interval: integer, interval of wrap around in units of nf * df (bandwidth)
        nwraps: number of wraps to include.
        no_regularization: bool,  if True, do not include diagonal regularization.

    Returns
    ----------
     (nchan, nchan) complex inverse of the tophat filtering matrix assuming that the delay-space covariance is diagonal and zero outside
         of the horizon
    """
    if cache is None:
        cache = {}
    if isinstance(filter_factors,(float,int, np.int, np.float)):
        filter_factors = [filter_factors]
    if isinstance(filter_centers, (float, int, np.int, np.float)):
        filter_centers = [filter_centers]
    if isinstance(filter_half_widths, (float, int, np.int, np.float)):
        filter_half_widths = [filter_half_widths]

    nchan = len(x)

    filter_key = _fourier_filter_hash(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                         filter_factors=filter_factors, x=x, w=None, hash_decimal=hash_decimal,
                                         label='dayenu_matrix_inverse', wrap=wrap, wrap_interval=wrap_interval,
                                         nwraps=nwraps, no_regularization=no_regularization)
    if not filter_key in cache:
        fx, fy = np.meshgrid(x,x)
        sdwi_mat = np.identity(fx.shape[0]).astype(np.complex128)
        if no_regularization:
            sdwi_mat *= 0.
        for fc, fw, ff in zip(filter_centers, filter_half_widths, filter_factors):
            if not ff == 0:
                if not wrap:
                    sdwi_mat = sdwi_mat + np.sinc( 2. * (fx-fy) * fw ).astype(np.complex128)\
                            * np.exp(-2j * np.pi * (fx-fy) * fc) / ff
                else:
                    bwidth = x[-1] - x[0] + (x[1]-x[0])
                    for wnum in np.arange(-nwraps//2, nwraps//2):
                        offset = bwidth * wnum * wrap_interval
                        sdwi_mat = sdwi_mat + \
                        np.sinc( 2. *  (fx-fy - offset) * fw  ).astype(np.complex128)\
                        * np.exp(-2j * np.pi * (fx-fy - offset) * fc) / ff
        cache[filter_key] = sdwi_mat
    else:
        sdwi_mat = cache[filter_key]
    return sdwi_mat
