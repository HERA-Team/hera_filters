from .. import dspec
import numpy as np, random
import pytest
from pyuvdata import UVData
from uvtools.data import DATA_PATH
import os
import scipy.signal.windows as windows
import warnings
random.seed(0)
import copy

def test_wedge_width():
    # Test boundaries of delay bins
    assert dspec.wedge_width(0, .01, 10) == (1,10)
    assert dspec.wedge_width(5., .01, 10) == (1,10)
    assert dspec.wedge_width( 9., .01, 10) == (2,-1)
    assert dspec.wedge_width(10., .01, 10) == (2,-1)
    assert dspec.wedge_width(15., .01, 10) == (3,-2)
    # test nchan
    assert dspec.wedge_width(10., .01, 20) == (3,-2)
    assert dspec.wedge_width(10., .01, 40) == (5,-4)
    # test sdf
    assert dspec.wedge_width(10., .02, 10) == (3,-2)
    assert dspec.wedge_width(10., .04, 10) == (5,-4)
    # test standoff
    assert dspec.wedge_width(100., .001, 100, standoff=4.) == (11,-10)
    assert dspec.wedge_width(100., .001, 100, standoff=5.) == (11,-10)
    assert dspec.wedge_width(100., .001, 100, standoff=10.) == (12,-11)
    assert dspec.wedge_width(100., .001, 100, standoff=15.) == (13,-12)
    # test horizon
    assert dspec.wedge_width(100., .001, 100, horizon=.1) == (2,-1)
    assert dspec.wedge_width(100., .001, 100, horizon=.5) == (6,-5)
    assert dspec.wedge_width(100., .001, 100, horizon=1.5) == (16,-15)
    assert dspec.wedge_width(100., .001, 100, horizon=2.) == (21,-20)

def test_delay_filter_dims():
    pytest.raises(ValueError, dspec.delay_filter, np.zeros((1,2,3)), np.zeros((1,2,3)), 0, .001)

def test_delay_filter_1D():
    NCHAN = 128
    TOL = 1e-6
    data = np.ones(NCHAN, dtype=np.complex)
    wgts = .5*np.ones(NCHAN, dtype=np.complex)
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
    np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
    np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
    wgts[::16] = 0
    # This test should have been failing since _w = 0.46 but skip_wgt=0.5 by default for delay_filter.
    # The reason it was not failing originally was because no 1d check for skip_wgt existed in high_pass_fourier_filter.
    # This check does exist in fourier_filter (as it should) and now the test, in its original form, fails.
    # I've changed the skip_wgt to 0.1 (down from 0.5) so that it passes.
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL, skip_wgt=0.1)
    np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
    np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
    data = np.random.normal(size=NCHAN)
    wgts = np.ones_like(data)
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=1e-9)
    assert np.allclose(np.average(data), np.average(dmdl), rtol=0, atol=1e-3)
    assert np.allclose(np.average(dres), 0.0, rtol=0, atol=1e-3)

    #check that skip_wgt is properly passed to clean
    wgts[:72] = 0.
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL, skip_wgt=0.5, mode='clean')
    assert info['status']['axis_1'][0] == 'skipped'

def test_delay_filter_2D():
    NCHAN = 128
    NTIMES = 10
    TOL = 1e-6
    data = np.ones((NTIMES, NCHAN), dtype=np.complex)
    wgts = np.ones((NTIMES, NCHAN), dtype=np.complex)
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
    np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
    np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
    wgts[:,::16] = 0;
    wgts*=.9 #tests to make sure wgts**2 normalization works
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL)
    np.testing.assert_allclose(data, dmdl, atol=NCHAN*TOL)
    np.testing.assert_allclose(dres, np.zeros_like(dres), atol=NCHAN*TOL)
    data = np.array(np.random.normal(size=(NTIMES,NCHAN)),dtype=complex)
    wgts = np.ones_like(data)
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=1e-9)
    np.testing.assert_allclose(np.average(data,axis=1), np.average(dmdl,axis=1), atol=1e-3)
    np.testing.assert_allclose(np.average(dres,axis=1), 0, atol=1e-3)
    #check that skip_wgt is properly passed to clean
    wgts[0,:72] = 0.
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL, skip_wgt=0.5, mode='clean')
    assert info['status']['axis_1'][0] == 'skipped'
    assert info['status']['axis_1'][1] == 'success'

def test_fourier_model():
    NMAX = 7
    NFREQS = 100
    nmodes = 2*NMAX + 1
    cn = (np.arange(nmodes) + 1.j*np.arange(nmodes)) / float(nmodes)
    model = dspec.fourier_model(cn, NFREQS)

    # Test shape of output model
    assert (NFREQS,) == model.shape

    # Test errors
    pytest.raises(ValueError, dspec.fourier_model, 3, NFREQS)
    pytest.raises(ValueError, dspec.fourier_model, np.empty((3, 3)), NFREQS)

def test_delay_filter_leastsq():
    NCHAN = 128
    NTIMES = 10
    TOL = 1e-7
    data = np.ones((NTIMES, NCHAN), dtype=np.complex)
    flags = np.zeros((NTIMES, NCHAN), dtype=np.bool)
    sigma = 0.1 # Noise level (not important here)

    # Fourier coeffs for input data, ordered from (-nmax, nmax)
    cn = np.array([-0.1-0.1j, -0.1+0.1j, -0.3-0.01j, 0.5+0.01j,
                   -0.3-0.01j, -0.1+0.1j, 0.1-0.1j])
    data *= np.atleast_2d( dspec.fourier_model(cn, NCHAN) )

    # Estimate smooth Fourier model on unflagged data
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq(data, flags,
                                                            sigma, nmax=3,
                                                            add_noise=False)
    np.testing.assert_allclose(data, bf_model, atol=NCHAN*TOL)
    np.testing.assert_allclose(cn, cn_out[0], atol=1e-6)

    # Estimate smooth Fourier model on data with some flags
    flags[:,10] = True
    flags[:,65:70] = True
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq(data, flags,
                                                            sigma, nmax=3,
                                                            add_noise=False)
    np.testing.assert_allclose(data, bf_model, atol=NCHAN*TOL)
    np.testing.assert_allclose(data, data_out, atol=NCHAN*TOL)

    # Test 1D code directly
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
        data[0], flags[0], sigma, nmax=3, add_noise=False)
    np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)

    # Test 1D code with non-linear leastsq
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
        data[0], flags[0], sigma, nmax=3, add_noise=False, use_linear=False)
    np.testing.assert_allclose(data[0], bf_model, atol=NCHAN*TOL)

    # Test that noise injection can be switched on
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
        data[0], flags[0], sigma, nmax=3, add_noise=True)
    np.testing.assert_allclose(data[0], bf_model, atol=NCHAN * TOL * sigma)

    # Test with a noise array
    sigma_array = sigma * np.ones_like(data[0])
    bf_model, cn_out, data_out = dspec.delay_filter_leastsq_1d(
        data[0], flags[0], sigma_array, nmax=3, add_noise=True)
    np.testing.assert_allclose(data[0], bf_model, atol=NCHAN * TOL * sigma)

    # Test errors
    pytest.raises(ValueError, dspec.delay_filter_leastsq_1d,
                     data[0], flags[0], sigma, nmax=3, operator=np.empty((3, 3)))
    pytest.raises(ValueError, dspec.delay_filter_leastsq_1d,
                     data[0], flags[0], sigma, nmax=3, cn_guess=np.array([3]))

def test_skip_wgt():
    NCHAN = 128
    NTIMES = 10
    TOL = 1e-6
    data = np.ones((NTIMES, NCHAN), dtype=np.complex)
    wgts = np.ones((NTIMES, NCHAN), dtype=np.complex)
    wgts[0, 0:-4] = 0
    dmdl, dres, info = dspec.delay_filter(data, wgts, 0., .1/NCHAN, tol=TOL, skip_wgt=.1)
    np.testing.assert_allclose(data[1:,:], dmdl[1:,:], atol=NCHAN*TOL)
    np.testing.assert_allclose(dres[1:,:], np.zeros_like(dres)[1:,:], atol=NCHAN*TOL)
    np.testing.assert_allclose(dmdl[0,:], np.zeros_like(dmdl[0,:]), atol=NCHAN*TOL)
    np.testing.assert_allclose(dres[0,:], (data * wgts)[0,:], atol=NCHAN*TOL)
    assert len(info['status']['axis_1']) == NTIMES
    assert np.all([info['status']['axis_1'][i] == 'skipped' for i in list(info['status']['axis_1'])[:1]])
    assert not np.any([info['status']['axis_1'][i] == 'skipped' for i in list(info['status']['axis_1'])[1:]])

def test_calc_width():
    # test single filter_size
    nchan = 100
    dt = 10.
    filter_size = 1e-2
    u, l = dspec.calc_width(filter_size, dt, nchan)
    frs = np.fft.fftfreq(nchan, dt)  # negative b/c of ifft convention
    assert np.all(np.abs(frs[u:l]) > filter_size)

    # test multiple entries in filter_size
    filter_size = (1e-2, 2e-2)
    u, l = dspec.calc_width(filter_size, dt, nchan)
    assert np.all((frs[u:l] < -1e-2) | (frs[u:l] > 2e-2))

def test_gen_window():
    for w in ['none', 'blackmanharris', 'hann', 'tukey', 'barthann', 'blackmanharris-7term',
              'cosinesum-9term', 'cosinesum-11term']:
        win = dspec.gen_window(w, 100)
        assert len(win) == 100
        assert isinstance(win, np.ndarray)
        assert win.min() >= 0.0
        assert win.max() <= 1.0
        pytest.raises(ValueError, dspec.gen_window, w, 100, normalization='foo')
        win2 = dspec.gen_window(w, 100,normalization='mean')
        assert np.all(np.isclose(win, win2*np.mean(win),atol=1e-6))
        win3 = dspec.gen_window(w, 100,normalization='rms')
        assert np.all(np.isclose(win, win3*np.sqrt(np.mean(win**2.)),atol=1e-6))

    pytest.raises(ValueError, dspec.gen_window, 'foo', 200)
    # check Ncut ValueError
    pytest.raises(ValueError, dspec.gen_window, 'bh', 200, edgecut_hi=101, edgecut_low=100)


def test_dft_operator():
    NF = 100
    DF = 100e3
    freqs = np.arange(-NF/2, NF/2)*DF + 150e6
    #test dft_operator by checking whether
    #it gives us expected values.
    fop = dspec.dft_operator(freqs, 0., 1e-6)
    fg, dg = np.meshgrid(freqs-150e6, np.arange(-10, 10) * (1./DF/NF) , indexing='ij')
    y = np.exp(2j * np.pi * fg * dg )
    np.testing.assert_allclose(fop, y)
    fg, dg = np.meshgrid(freqs-150e6, np.arange(-20, 20) * (1./DF/NF/2) , indexing='ij')
    #check fundamental period x 2 works alright
    #and gives us expected values
    y1 = np.exp(2j * np.pi * fg * dg )
    fop1 = dspec.dft_operator(freqs, 0., 1e-6, fundamental_period=200*1e5)
    np.testing.assert_allclose(fop1, y1)


def test_dpss_operator():
    #test that an error is thrown when we specify more then one
    #termination method.
    NF = 100
    DF = 100e3
    freqs = np.arange(-NF/2, NF/2)*DF + 150e6
    freqs_bad = np.array([1.100912386458, 1.22317, 2.12341260, 3.234632462, 5.32348356887])
    pytest.raises(ValueError, dspec.dpss_operator, x=freqs_bad, filter_centers=[0.], filter_half_widths=[1e-6], nterms=[5])
    pytest.raises(ValueError, dspec.dpss_operator, x=freqs , filter_centers=[0.], filter_half_widths=[1e-6], nterms=[5], avg_suppression=[1e-12])
    #now calculate DPSS operator matrices using different cutoff criteria. The columns
    #should be the same up to the minimum number of columns of the three techniques.
    amat1, ncol1 = dspec.dpss_operator(freqs, [0.], [100e-9], eigenval_cutoff=[1e-9])
    amat2, ncol2 = dspec.dpss_operator(freqs, [0.], [100e-9], edge_suppression=[1e-9])
    amat3, ncol3 = dspec.dpss_operator(freqs, [0.], [100e-9], avg_suppression=[1e-9])
    ncols = [ncol1, ncol2, ncol3]
    ncolmin = np.min(ncols)
    ncolmax = np.max(ncols)
    amat4, ncol4 = dspec.dpss_operator(freqs, [0.], [100e-9], nterms=[ncolmax])
    assert ncol4[0]==ncolmax
    #check that all columns of matrices obtained with different methods
    #of cutoff are identical.
    for m in range(ncolmin):
        np.testing.assert_allclose(amat1[:,m], amat2[:,m])
        np.testing.assert_allclose(amat2[:,m], amat3[:,m])
        np.testing.assert_allclose(amat3[:,m], amat4[:,m])

    dpss_mat = windows.dpss(NF, NF * DF * 100e-9, ncolmax).T
    for m in range(ncolmax):
        np.testing.assert_allclose(amat4[:,m], dpss_mat[:,m])


def test_fit_solution_matrix():
    #test for dft and dpss
    fs = np.arange(-50,50)
    #here is some underlying data
    data = np.exp(2j * np.pi * 3.5/50. * fs) + 5*np.exp(2j * np.pi * 2.1/50. * fs)
    data += np.exp(-2j * np.pi * 0.7/50. * fs) + 5*np.exp(2j * np.pi * -1.36/50. * fs)
    #here are some weights with flags
    wgts = np.ones_like(data)
    wgts[6] = 0
    wgts[17] = 0
    dw = data*wgts
    #generate fits for DPSS and DFT.
    amat_dft = dspec.dft_operator(fs, [0.], [4. / 50.], fundamental_period=140.)
    amat_dpss,_ = dspec.dpss_operator(fs, [0.], [5. / 50.], eigenval_cutoff=[1e-15])
    wmat = np.diag(wgts)
    fitmat_dft = dspec.fit_solution_matrix(wmat, amat_dft)
    fitmat_dpss = dspec.fit_solution_matrix(wmat, amat_dpss)
    interp_dft = amat_dft @ fitmat_dft @ dw
    interp_dpss = amat_dpss @ fitmat_dpss @ dw
    #DFT interpolation is meh, so we keep our standards low.
    #DFT interpolation matrices are poorly conditioned so that's also
    #Downer.
    assert np.all(np.isclose(interp_dft, data, atol=1e-2))
    #DPSS interpolation is clutch. We can make our standards high.
    assert np.all(np.isclose(interp_dpss, data, atol=1e-6))
    #Check Raising of ValueErrors.
    amat_dft_pc = dspec.dft_operator(fs, [0.], [4. / 50.], fundamental_period=200.)
    with warnings.catch_warnings(record=True) as w:
        dspec.fit_solution_matrix(wmat, amat_dft_pc)
        assert len(w) > 0
    pytest.raises(ValueError, dspec.fit_solution_matrix, wmat[:50], amat_dft_pc)
    pytest.raises(ValueError, dspec.fit_solution_matrix, wmat, amat_dft[:-1])


def test_dayenu_filter():
    nf = 100
    df = 100e3
    freqs = np.arange(-nf//2, nf//2) * df
    #generate some noise
    noise = (np.random.randn(nf) + 1j * np.random.randn(nf))/np.sqrt(2.)
    #a foreground tone and a signal tone
    fg_tone = 1e4 * np.exp(2j * np.pi * freqs * 50e-9)
    sg_tone = 1e2 * np.exp(2j * np.pi * freqs * 1000e-9)
    fg_ns = noise + fg_tone
    fg_sg =  fg_tone + sg_tone
    data_1d = fg_ns
    data_2d = np.array([data_1d, data_1d])
    filter_centers = [0.]
    filter_half_widths = [200e-9]
    filter_factors = [1e-9]
    wghts_1d = np.ones(nf)
    wghts_2d = np.array([wghts_1d, wghts_1d])
    #test functionality for numpy arrays
    dspec.dayenu_filter(np.arange(-nf/2,nf/2)*df, data_1d, wghts_1d, [1], np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors))
    #provide filter_dimensions as an integer.
    dspec.dayenu_filter(np.arange(-nf/2,nf/2)*df, data_1d, wghts_1d, 1, np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors))

    #test functionality on floats
    dspec.dayenu_filter(np.arange(-nf/2,nf/2)*df, data_1d, wghts_1d,  1, filter_centers[0], filter_half_widths[0],
                        filter_factors[0])
    filter_half_widths2 = [200e-9, 200e-9]
    filter_centers2 = [0., -1400e-9]
    filter_factors2 = [1e-9, 1e-9]
    #check if throws error when number of filter_half_widths not equal to len filter_centers
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, data_1d, wghts_1d, [1], filter_centers,
                    filter_half_widths2, filter_factors)
    #check if throws error when number of filter_half_widths not equal to len filter_factors
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, data_1d, wghts_1d, 1, filter_centers,
                    filter_half_widths, filter_factors2)
    #check if error thrown when wghts have different length then data
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, data_1d, wghts_1d[:-1], 1, filter_centers,
                    filter_half_widths, filter_factors)
    #check if error thrown when dimension of data does not equal dimension of weights.
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, data_1d, wghts_2d, 1, filter_centers,
                    filter_half_widths, filter_factors)
    #check if error thrown if dimension of data does not equal 2 or 1.
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, np.zeros((10,10,10)), wghts_1d, 1, filter_centers,
                    filter_half_widths, filter_factors)
    #check if error thrown if dimension of weights does not equal 2 or 1.
    pytest.raises(ValueError, dspec.dayenu_filter, freqs, wghts_1d, np.zeros((10,10,10)), 1, filter_centers,
                    filter_half_widths, filter_factors)
    # check error where x is not a numpy array
    pytest.raises(ValueError, dspec.dayenu_filter, x='x', data=data_1d, wgts=wghts_1d, filter_dimensions=[1], filter_centers=filter_centers,
                     filter_half_widths=filter_half_widths,
                     filter_factors=filter_factors)
    # check error where filter-dimensions is not an integer or tuple/list
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_1d, wgts=wghts_1d, filter_dimensions='[1]',
                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                     filter_factors=filter_factors)
    # check lenght of filter_dims is > 2
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_1d, wgts=wghts_1d, filter_dimensions=[0, 1, 2],
                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                     filter_factors=filter_factors)
    # check that filter_dimensions are integers.
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_1d, wgts=wghts_1d, filter_dimensions=[0.0],
                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                     filter_factors=filter_factors)
    # check filter dimensions are either 0 or 1.
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_1d, wgts=wghts_1d, filter_dimensions=[2],
                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                     filter_factors=filter_factors)
    # check error if negative filter_factor provided
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_1d, wgts=wghts_1d, filter_dimensions=[0],
                     filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                     filter_factors=[-1e-9])
    #now filter foregrounds and test that std of residuals are close to std of noise:
    filtered_noise, _ =  dspec.dayenu_filter(np.arange(-nf/2, nf/2)*df, data_1d, wghts_1d, [1], filter_centers, filter_half_widths,
                                         filter_factors)
    #print(np.std((data_1d - fg_tone).real)*np.sqrt(2.))
    #print(np.std((filtered_noise).real)*np.sqrt(2.))
    np.testing.assert_almost_equal( np.std(filtered_noise.real)**2. + np.std(filtered_noise.imag)**2.,
                                  np.std(noise.real)**2. + np.std(noise.imag)**2., decimal = 0)
    #now filter foregrounds and signal and test that std of residuals are close to std of signal.
    filtered_signal, _ =  dspec.dayenu_filter(np.arange(-nf/2, nf/2)*df, fg_sg, wghts_1d, [1], filter_centers, filter_half_widths,
                                              filter_factors)

    np.testing.assert_almost_equal( (np.std(filtered_signal.real)**2. + np.std(filtered_signal.imag)**2.)/1e4,
                                  (np.std(sg_tone.real)**2. + np.std(sg_tone.imag)**2.)/1e4, decimal = 0)

    # check that zeros where wgts are zero.
    wghts_1d[len(wghts_1d)//4] = 0.
    wghts_1d[len(wghts_1d)//4 + 5] = 0.
    filtered_noise, _ =  dspec.dayenu_filter(np.arange(-nf/2, nf/2)*df, data_1d, wghts_1d, [1], filter_centers, filter_half_widths,
                                            filter_factors)
    assert np.all(filtered_noise[~(wghts_1d.astype(bool))] == 0.)



    #Next, we test performing a fringe-rate clean. Generate a 50-meter EW baseline with a single
    #source moving overhead perpindicular to baseline
    TEST_CACHE = {}
    OMEGA_EARTH = 2. * np.pi / 3600. / 24.
    times = np.linspace(-1800, 1800., nf, endpoint = False)
    dt = times[1]-times[0]
    freqs = np.linspace(145e6, 155e6, nf, endpoint=False)
    fg, tg = np.meshgrid(freqs,times)
    signal_2d = 1e6 * np.exp(2j * np.pi * 50. / 3e8 * np.sin(OMEGA_EARTH * tg) * fg)
    noise_2d = np.random.randn(nf,nf)/np.sqrt(2.)\
    + 1j*np.random.randn(nf,nf)/np.sqrt(2.)
    data_2d = signal_2d + noise_2d

    # check that if we are performing 2d filtering, then x is a length 2 list.
    pytest.raises(ValueError, dspec.dayenu_filter, x=np.arange(-nf/2, nf/2)*df, data=data_2d, wgts=np.ones_like(data_2d), filter_dimensions=[1, 0],
                     filter_centers=[[0.],[0.]], filter_half_widths=[[1e-3], [100e-9]],
                     filter_factors=[[1e-9], [1e-9]])
    # check that if we are performing 2d filtering, then x is a length 2 list and each x is a numpy array, list, or tuple.
    pytest.raises(ValueError, dspec.dayenu_filter, x=['time is a construct', np.arange(-nf/2, nf/2)*df], data=data_2d, wgts=np.ones_like(data_2d), filter_dimensions=[1, 0],
                     filter_centers=[[0.],[0.]], filter_half_widths=[[1e-3], [100e-9]],
                     filter_factors=[[1e-9], [1e-9]])
    # check value error if length of one of the filter_centers is greater then 2
    pytest.raises(ValueError, dspec.dayenu_filter, x=[times, np.arange(-nf/2, nf/2)*df], data=data_2d, wgts=np.ones_like(data_2d), filter_dimensions=[1, 0],
                     filter_centers=[[0.], [0.]], filter_half_widths=[[1e-3], [100e-9]],
                     filter_factors=[[1e-9], [1e-9], [1e-9]])

    #now, only filter fringe-rate domain. The fringe rate for a source
    #overhead should be roughly 0.0036 for this baseline.
    filtered_data_fr, _ = dspec.dayenu_filter(np.arange(-nf/2,nf/2)*dt, data_2d, np.ones_like(data_2d),
                        filter_centers = [0.], filter_half_widths = [0.004], filter_factors = [1e-10],
                        filter_dimensions = [0], cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_fr.flatten())**2.)),
                                    1., decimal = 1)

    #only filter in the delay-domain.

    filtered_data_df, _ = dspec.dayenu_filter(np.arange(-nf/2, nf/2)*df, data_2d, np.ones_like(data_2d),
                        filter_centers = [0.], filter_half_widths=[100e-9], filter_factors=[1e-10],
                        filter_dimensions = [1], cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df.flatten())**2.)),
                                                            1., decimal = 1)
    #supply an int for filter dimensions
    filtered_data_df, _ = dspec.dayenu_filter(np.arange(-nf/2, nf/2)*df, data_2d, np.ones_like(data_2d),
                        filter_centers = [0.], filter_half_widths=[100e-9], filter_factors=[1e-10],
                        filter_dimensions = 1, cache = TEST_CACHE)

    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df.flatten())**2.)),
                                    1., decimal = 1)

    #filter in both domains. I use a smaller filter factor
    #for each domain since they multiply in the target region.

    filtered_data_df_fr, _ = dspec.dayenu_filter([np.arange(-nf/2,nf/2)*dt, np.arange(-nf/2,nf/2)*df],
                                                    data_2d, np.ones_like(data_2d),
                                                    filter_centers = [[0.002],[0.]],
                                                    filter_half_widths = [[0.001],[100e-9]], filter_factors = [[1e-5],[1e-5]],
                                                    filter_dimensions = [0,1],cache = TEST_CACHE)

    # check for zero residual in flagged regions
    wgts_2d = np.ones_like(data_2d)
    wgts_2d[nf // 2+5]  = 0.
    wgts_2d[:, nf // 4 + 3] = 0.
    filtered_data_df_fr, _ = dspec.dayenu_filter([np.arange(-nf/2,nf/2)*dt, np.arange(-nf/2,nf/2)*df],
                                                    data_2d, wgts_2d,
                                                    filter_centers = [[0.002],[0.]],
                                                    filter_half_widths = [[0.001],[100e-9]], filter_factors = [[1e-5],[1e-5]],
                                                    filter_dimensions = [0,1],cache = TEST_CACHE)
    assert np.all(filtered_data_df_fr[:, nf // 4 + 3] == 0)
    assert np.all(filtered_data_df_fr[nf // 2 + 5] == 0)
    np.testing.assert_almost_equal(np.sqrt(np.mean(np.abs(filtered_data_df_fr.flatten())**2.)),
                                    1., decimal = 1)

    #test error messages if we do not provide lists of lists.
    pytest.raises(ValueError,dspec.dayenu_filter,[np.arange(-nf/2,nf/2)*dt, np.arange(-nf/2,nf/2)*df],
                        data_2d, np.ones_like(data_2d),
                        filter_centers = [[0.002],0.],
                        filter_half_widths = [[0.001],[100e-9]],
                        filter_factors = [1e-5,[1e-5]],
                        filter_dimensions = [0,1],cache = {})

    # test skip_wgt:
    _, info = dspec.dayenu_filter(np.arange(-nf/2,nf/2)*df, data_1d, np.zeros_like(wghts_1d), [1], np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors))
    assert np.all([info['status']['axis_1'][i] == 'skipped' for i in info['status']['axis_1']])

    _, info = dspec.dayenu_filter(np.arange(-nf/2,nf/2)*df, data_1d, np.ones_like(wghts_1d), [1], np.array(filter_centers), np.array(filter_half_widths),
                        np.array(filter_factors))
    assert np.all([info['status']['axis_1'][i] == 'success' for i in info['status']['axis_1']])


def test_dayenu_mat_inv():
    freqs = np.arange(-16,16)*100e3
    cmat = dspec.dayenu_mat_inv(freqs, filter_centers = [], filter_half_widths = [], filter_factors = [])
    #verify that the inverse cleaning matrix without cleaning windows is the identity!
    np.testing.assert_array_equal(cmat, np.identity(32).astype(np.complex128))
    #next, test with a single filter window with list and float arguments supplied
    cmat1 = dspec.dayenu_mat_inv(freqs, filter_centers = 0., filter_half_widths = 112e-9, filter_factors = 1e-9)
    cmat2 = dspec.dayenu_mat_inv(freqs, filter_centers = [0.], filter_half_widths = [112e-9], filter_factors = [1e-9])
    x,y = np.meshgrid(np.arange(-16,16), np.arange(-16,16))
    cmata = np.identity(32).astype(np.complex128) + 1e9 * np.sinc( (x-y) * 100e3 * 224e-9 ).astype(np.complex128)
    np.testing.assert_array_equal(cmat1, cmat2)
    #next test that the array is equal to what we expect
    np.testing.assert_almost_equal(cmat1, cmata)
    #now test no_regularization
    cmat1 = dspec.dayenu_mat_inv(freqs, filter_centers = 0.,
            filter_half_widths = 112e-9, filter_factors = 1, no_regularization = True)
    np.sinc( (x-y) * 100e3 * 224e-9 ).astype(np.complex128)
    np.testing.assert_almost_equal(cmat1, cmata / 1e9)
    #now test wrap!
    cmat1 = dspec.dayenu_mat_inv(freqs, filter_centers = 0.,
            filter_half_widths = 112e-9, filter_factors = 1, wrap = True,
             no_regularization = True)
    cmata = np.zeros_like(cmat1)
    for m in range(-500,500):
        cmata += np.sinc((x-y - 32 * m) * 100e3 * 224e-9)
    np.testing.assert_almost_equal(cmat1, cmata)


def test_vis_filter():
    # load file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])

    freqs = uvd.freq_array.squeeze()
    times = np.unique(uvd.time_array) * 24 * 3600
    times -= np.mean(times)
    sdf = np.median(np.diff(freqs))
    dt = np.median(np.diff(times))
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9

    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[2]+ddly/4)/1e9)
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[20]) - 2j*np.pi*freqs[None, :]*(dlys[20])/1e9)
    d += 10 * ((np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs))

    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.median(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # get snr of modes
    freq_snr1, freq_snr2 = get_snr(d, fftax=1, avgax=0, modes=[2, 20])
    time_snr1, time_snr2 = get_snr(d, fftax=0, avgax=1, modes=[2, 20])
    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = 70.0 / 2.99e8

    # try passing skip_wgt
    mdl, res, info = dspec.delay_filter(d, w, sdf, bl_len,
                                        tol=1e-4, window='none', skip_wgt=1. - 1. / f.shape[1], gain=0.1)
    # with this skip_wgt, all times should be skipped.
    assert np.all([info['status']['axis_1'][i] == 'skipped' for i in info['status']['axis_1'] ])

    # delay filter basic execution
    mdl, res, info = dspec.delay_filter(d, w, sdf, bl_len,
                                        tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)

    #check skips .
    assert info['status']['axis_1'][20] == 'skipped'
    assert np.all([info['status']['axis_1'][i] == 'success' for i in info['status']['axis_1'] if i != 20])

    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    assert np.isclose(snrs[0], freq_snr1, atol=3)
    assert np.isclose(snrs[1], freq_snr2, atol=3)

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                               tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    assert np.isclose(mdl - mdl2, 0.0).all()

    # fringe filter basic execution
    mdl, res, info = dspec.fringe_filter(d, w, frs[15], dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    assert np.isclose(snrs[0], time_snr1, atol=3)
    assert np.isclose(snrs[1], time_snr2, atol=3)

    # test vis filter is the same
    mdl2, res2, info2 = dspec.vis_filter(d, w, max_frate=frs[15], dt=dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln2 = mdl2 + res2
    assert np.isclose(mdl - mdl2, 0.0).all()

    # try non-symmetric filter
    mdl, res, info = dspec.fringe_filter(d, w, (frs[-20], frs[10]), dt, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    cln = mdl + res

    assert np.all([info['status']['axis_1'][i] == 'success' for i in info['status']['axis_1'] if i != 20])

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=0, avgax=1)
    assert np.isclose(snrs[0], time_snr1, atol=3)
    assert np.isclose(snrs[1], time_snr2, atol=3)

    # 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)
    cln = mdl + res
    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    assert np.isclose(snrs[0], freq_snr1, atol=3)
    assert np.isclose(snrs[1], freq_snr2, atol=3)

    # non-symmetric 2D clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, tol=1e-4, window='none', maxiter=100, gain=1e-1)
    cln = mdl + res

    # assert recovered snr of input modes
    snrs = get_snr(cln, fftax=1, avgax=0)
    assert np.isclose(snrs[0], freq_snr1, atol=3)
    assert np.isclose(snrs[1], freq_snr2, atol=3)

    # try plus filtmode on 2d clean
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[10], frs[10]), dt=dt, tol=1e-4, window=('none', 'none'), edgecut_low=(0, 5), edgecut_hi=(2, 5), maxiter=100, gain=1e-1, filt2d_mode='plus')
    mfft = np.fft.ifft2(mdl)
    cln = mdl + res

    # assert clean components fall only in plus area
    clean_comp = np.where(~np.isclose(np.abs(mfft), 0.0))
    for cc in zip(*clean_comp):
        assert 0 in cc

    # exceptions
    pytest.raises(ValueError, dspec.vis_filter, d, w, bl_len=bl_len, sdf=sdf, max_frate=(frs[-20], frs[10]), dt=dt, filt2d_mode='foo')

    # test add_clean_residual: test res of filtered modes are lower when add_residual is True
    mdl, res, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-6, window='none', maxiter=100, gain=1e-1, add_clean_residual=False)
    mdl2, res2, info = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, max_frate=frs[15], dt=dt, tol=1e-6, window='none', maxiter=100, gain=1e-1, add_clean_residual=True)
    rfft = np.fft.ifft2(res)
    rfft2 = np.fft.ifft2(res2)
    assert np.median(np.abs(rfft2[:15, :23] / rfft[:15, :23])) < 1

def test_delay_interpolation_matrix():
    """
    Test inerpolation matrix.

    Test creates some simple data with a few underlying delays.
    Flagged data is generated from underlying data.
    The flagged channels are interpolated over using the delay_interpolation_matrix
    and the interpolation is compared to the original.

    """
    MYCACHE={}
    fs = np.arange(-10,10)
    #here is some underlying data
    data = np.exp(2j * np.pi * 3/20. * fs) + 5*np.exp(2j * np.pi * 4./20 * fs)
    data += np.exp(-2j * np.pi * 3/20. * fs) + 5*np.exp(2j * np.pi * -4/20. * fs)
    #here are some weights with flags
    wgts = np.ones_like(data)
    wgts[6] = 0
    wgts[17] = 0
    dw = data*wgts
    #here is some flagged data.
    #interpolate data and see if it matches true data.
    data_interp = np.dot(dspec.delay_interpolation_matrix(nchan=20, ndelay=5, wgts=wgts, fundamental_period=20, cache=MYCACHE), dw)
    #check that interpolated data agrees with original data.
    assert  np.all(np.isclose(data_interp, data, atol=1e-6))
    #test error raising.
    pytest.raises(ValueError, dspec.delay_interpolation_matrix, 10, 2, np.ones(5))
    pytest.raises(ValueError, dspec.delay_interpolation_matrix, 5, 2, np.asarray([0., 0., 0., 0., 0.]))
    #test diagnostic mode.
    data_interp1 = dspec.delay_interpolation_matrix(nchan=20, ndelay=5, wgts=wgts,
     fundamental_period=20, cache={})
    data_interp1 = np.dot(data_interp1, dw)
    assert np.all(np.isclose(data_interp, data_interp1, atol=1e-6))
     #test warning
    with warnings.catch_warnings(record=True) as w:
        wgtpc = np.ones(100)
        randflags = np.asarray([29, 49,  6, 47, 68, 98, 69, 70, 32,  3]).astype(int)
        wgtpc[randflags]=0.
        amat_pc = dspec.delay_interpolation_matrix(nchan=100, ndelay=25, wgts=wgtpc, fundamental_period=200)
        print(len(w))
        assert len(w) > 0

def test_fourier_filter():
    # load file
    dt = 10.7374267578125
    sdf = 100e3
    nf = 100
    ntimes = 120
    freqs = np.arange(-nf/2, nf/2) * sdf + 150e6
    times = np.arange(-ntimes/2, ntimes/2) * dt
    frs = np.fft.fftshift(np.fft.fftfreq(ntimes, d=dt))
    dfr = np.mean(np.diff(frs))
    dlys = np.fft.fftshift(np.fft.fftfreq(nf, d=sdf))
    ddly = np.mean(np.diff(dlys))
    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[ntimes//2+2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[nf//2+2]+ddly/4))
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[ntimes//2-3]) - 2j*np.pi*freqs[None, :]*(dlys[nf//2+3]))

    def get_snr(clean, fftax=1, avgax=0, modes=[2, 20]):
        cfft = np.fft.ifft(clean, axis=fftax)
        cavg = np.median(np.abs(cfft), axis=avgax)
        std = np.median(cavg)
        return [cavg[m] / std for m in modes]

    # get snr of modes
    freq_snr1, freq_snr2 = get_snr(d, fftax=1, avgax=0, modes=[2, 20])
    time_snr1, time_snr2 = get_snr(d, fftax=0, avgax=1, modes=[2, 20])

    # simulate some flags
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = dlys[nf//2+4]
    fr_len = frs[ntimes//2+4]
    # dpss filtering
    dpss_options1={'eigenval_cutoff':[1e-6]}
    dft_options1={'fundamental_period':2.*(times.max()-times.min())}
    dft_options2={'fundamental_period':2.*(times.max()-times.min())}
    dft_options3={'fundamental_period':2.*(dlys.max()-dlys.min())}
    clean_options1={'tol':1e-9, 'maxiter':100, 'filt2d_mode':'rect',
                    'edgecut_low':0, 'edgecut_hi':0, 'add_clean_residual':False,
                    'window':'none', 'gain':0.1, 'alpha':0.5}
    mdl1, res1, info1 = dspec.fourier_filter(x=freqs, data=d, wgts=w, filter_centers=[0.], cache={},
                                             filter_half_widths=[bl_len], suppression_factors=[0.],
                                             mode='dpss_leastsq', **dpss_options1)

    # check filter dims error is raised
    pytest.raises(ValueError, dspec.fourier_filter,x=freqs, data=d, wgts=w, filter_centers=[0.], filter_dims=2,
                                             filter_half_widths=[bl_len], suppression_factors=[0.],
                                             mode='dpss_leastsq', **dpss_options1)
    #check that length >2 filter dims will fail.
    pytest.raises(ValueError, dspec.fourier_filter,x=freqs, data=d, wgts=w, filter_centers=[0.], filter_dims=[0, 1, 1],
                                             filter_half_widths=[bl_len], suppression_factors=[0.],
                                             mode='dpss_leastsq', **dpss_options1)

    mdl2, res2, info2 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len], suppression_factors=[0.],
                                             mode='dpss_matrix', **dpss_options1)
    #check clean with and without default options gives equivalent answers.
    mdl3, res3, info3 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len],
                                             mode='clean')

    mdl4, res4, info4 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len],
                                             mode='clean', **clean_options1)


    clean_options_typo = {'tol':1e-9, 'maxiter':100, 'filt2d_mode':'rect',
                    'edgecut_low':0, 'edgecut_hi':0, 'add_clean_residual':False,
                    'window':'none', 'gain':0.1, 'alphae':0.5}
    #check that a ValueError is returned if we include a bad parameter name.
    pytest.raises(ValueError, dspec.fourier_filter, freqs, d, w, [0.], [bl_len],
                     mode='clean', **clean_options_typo)

    assert np.all(np.isclose(mdl3, mdl4, atol=1e-6))
    assert np.all(np.isclose(res3, res4, atol=1e-6))

    assert np.all(np.isclose(mdl1, mdl2, atol=1e-6))
    assert np.all(np.isclose(res1, res2))

    #check that dayenu can be run without fitting options.
    mdl3, res3, info3 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len], suppression_factors=[1e-9],
                                             mode='dayenu')

    mdl4, res4, info4 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len], suppression_factors=[1e-9],
                                             mode='dayenu')

    assert np.all(np.isclose(mdl3, mdl4, atol=1e-6))
    assert np.all(np.isclose(res3, res4, atol=1e-6))

    #check that clean skips if all data is equal to zero, avoids infinite loop case.
    mdl3, res3, info3 = dspec.fourier_filter(freqs, np.zeros_like(d), w, [0.], [bl_len],
                                             mode='clean', filter_dims=1)
    assert np.all([info3['status']['axis_1'][i] == 'skipped' for i in info3['status']['axis_1']])

    #check error when unsupported mode provided
    pytest.raises(ValueError, dspec.fourier_filter, x=freqs, data=d, wgts=w, filter_centers=[0.],
                    filter_half_widths=[bl_len], suppression_factors=[0.],
                    mode='foo', **dpss_options1)
    #check error when wgt dim does not equal data dim.
    pytest.raises(ValueError, dspec.fourier_filter, x=freqs, data=d, wgts=w[0].squeeze(), filter_centers=[0.],
                    filter_half_widths=[bl_len], suppression_factors=[0.],
                    mode='dpss_leastsq', **dpss_options1)

    #check 1d vector support
    mdl11d, res11d, info11d = dspec.fourier_filter(x=freqs, data=d[0], wgts=w[0], filter_centers=[0.],
                                             filter_half_widths=[bl_len], suppression_factors=[0.],
                                             mode='dpss_leastsq', **dpss_options1)

    #test that the info is properly switched. fourier_filter processes all data in frequency_mode and takes transposes for time
    #filtering mode.
    for k in info11d:
        assert len(info11d[k]['axis_0']) == 0
        if k == 'status':
            assert len(info11d[k]['axis_1']) == 1

    assert np.all(np.isclose(mdl1[0], mdl11d, atol=1e-6))
    #perform a fringe-rate filter
    mdl5, res5, info5 = dspec.fourier_filter(x=times, data=d, wgts=w, filter_centers=[0.],
                                             filter_half_widths=[fr_len], suppression_factors=[0.], filter_dims=0,
                                             mode='dpss_leastsq', **dpss_options1)

    #test that the info is properly switched. fourier_filter processes all data in frequency_mode and takes transposes for time
    #filtering mode.
    for k in info5:
        assert len(info5[k]['axis_1']) == 0
        if k == 'status':
            assert len(info5[k]['axis_0']) == d.shape[1]

    #check that fringe rate filter model gives similar results to delay filter.
    assert np.all(np.isclose(mdl1[~f],mdl5[~f], rtol=1e-2))

    #perform some sanity checks on handling of nans in dft_leastsq. If nans are present in fundamental period
    #then the default behavior should be to set fundamental period to 2 * bandwidth.
    mdl, res, info = dspec.fourier_filter(x=freqs, data=d[0], wgts=w[0], filter_centers=[0.],
                                             filter_half_widths=[bl_len], suppression_factors=[1e-9],
                                             mode='dft_leastsq')
    # check that the filter_period is indeed equal to 1 / (2 * bandwidth)
    assert np.isclose(info['filter_params']['axis_1']['basis_options']['fundamental_period'],
                      2 * (freqs.max() - freqs.min()))

    #check that user provided fundamental period agrees with whats in info.
    mdl, res, info = dspec.fourier_filter(x=freqs, data=d[0], wgts=w[0], filter_centers=[0.],
                                             filter_half_widths=[bl_len], suppression_factors=[1e-9],
                                             mode='dft_leastsq', fundamental_period=4. * (freqs.max() - freqs.min()))
    # check that the filter_period is indeed equal to 1 / (2 * bandwidth)
    assert np.isclose(info['filter_params']['axis_1']['basis_options']['fundamental_period'],
                      4. * (freqs.max() - freqs.min()))

    #check fringe rate filter with dft mode
    mdl6, res6, info6 = dspec.fourier_filter(x=times, data=d, wgts=w, filter_centers=[0.],
                                             filter_half_widths=[fr_len], suppression_factors=[0.], filter_dims=0,
                                             mode='dft_leastsq', **dft_options1)

    mdl62, res62, info62 = dspec.fourier_filter(x=times, data=d, wgts=w, filter_centers=[0.],
                                             filter_half_widths=[fr_len], suppression_factors=[0.], filter_dims=0,
                                             mode='dft_leastsq', **dft_options1)
    #check that dft and dpss fringe-rate inpainting give the same results.
    assert np.all(np.isclose(mdl5, mdl6, rtol=1e-2))
    assert np.all(np.isclose(mdl62, mdl6, rtol=1e-2))

    #Check Dayenu filter.
    mdl7, res7, info7 = dspec.fourier_filter(x=times, data=d, wgts=w, filter_centers=[0.],
                                             filter_half_widths=[fr_len], suppression_factors=[1e-8], filter_dims=0,
                                             mode='dayenu_dft_leastsq', **dft_options1)

    mdl8, res8, info8 = dspec.fourier_filter(x=times, data=d, wgts=w, filter_centers=[0.],
                                             filter_half_widths=[fr_len], suppression_factors=[1e-8], filter_dims=0,
                                             mode='dayenu_dpss_leastsq', **dpss_options1)
    assert np.all(np.isclose(mdl7, mdl8, rtol=1e-2))
    assert np.all(np.isclose(mdl5, mdl8, rtol=1e-2))

    for k in info8:
        if not k == 'info_deconv':
            assert len(info8[k]['axis_1']) == 0
            if k == 'status':
                assert len(info8[k]['axis_0']) == d.shape[1]
    for k in info8['info_deconv']:
            assert len(info8['info_deconv'][k]['axis_1']) == 0
            if k == 'status':
                assert len(info8['info_deconv'][k]['axis_0']) == d.shape[1]

    #perform 2d dayenu filter with dpss and dft deconvolution.
    dpss_options1_2d = {'eigenval_cutoff': [[1e-12], [1e-12]]}
    dft_options1_2d = {'fundamental_period': [np.nan, np.nan]}
    dft_options2_2d = {'fundamental_period': [4 * (times.max() - times.min()), 4 * (freqs.max() - freqs.min())]}

    mdl9, res9, info9 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[1e-8],[1e-8]],
                                             mode='dayenu_dpss_leastsq', filter_dims=[1, 0], **dpss_options1_2d)

    pytest.raises(ValueError, dspec.fourier_filter, x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[1e-8],[1e-8]],
                                             mode='dayenu_dpss_leastsq', filter_dims=[1, 0], **dpss_options1)

    pytest.raises(ValueError, dspec.fourier_filter, x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[1e-8],[1e-8]],
                                             mode='dayenu_dpss_leastsq', filter_dims=[1, 0], **dft_options1)

    mdl10, res10, info10 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[1e-8],[1e-8]],
                                             mode='dayenu_dft_leastsq', filter_dims=[1, 0], **dft_options1_2d)
   #check 2d filter dft fundamental period error.
    pytest.raises(ValueError, dspec.fourier_filter,x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[0.],[0.]],
                                             mode='dft_leastsq', filter_dims=[1, 0], **dpss_options1)
    mdl_dft, res_dft, info_dft = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[0.],[0.]],
                                             mode='dft_leastsq', filter_dims=[1, 0], **dft_options1_2d)

    mdl_dft1, res_dft1, info_dft1 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[0.],[0.]],
                                             mode='dft_leastsq', filter_dims=[1, 0])
    assert np.all(np.isclose(mdl_dft1, mdl_dft, rtol=1e-2))

    #try 2d iterative clean.
    mdl11, res11, info11 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]],
                                             mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'rect','tol':1e-5})
    #try out plus mode. IDK
    mdl12, res12, info12 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]],
                                             mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'plus','tol':1e-5})

    #make sure that clean is skipped if all weights are zero.
    mdl13, res13, info13 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=np.zeros_like(w), filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]],
                                             mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'plus','tol':1e-5})
    assert info13['clean_status']['axis_0']['skipped']
    assert info13['clean_status']['axis_1']['skipped']
    #test error when cleaning with invalid filt2d mode.
    pytest.raises(ValueError, dspec.fourier_filter,x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                                 filter_half_widths=[[fr_len],[bl_len]],
                                                 mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'bargh','tol':1e-5})
    #check equally spaced data value error for clean
    tlog = np.logspace(np.log10(times.min()), np.log10(times.max()), len(times))
    flog = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), len(freqs))
    pytest.raises(ValueError, dspec.fourier_filter,x=[tlog, flog], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                                 filter_half_widths=[[fr_len],[bl_len]],
                                                 mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'plus','tol':1e-5})
    pytest.raises(ValueError, dspec.fourier_filter,x=flog, data=d, wgts=w, filter_centers=[0.],
                                                  filter_half_widths=[bl_len],
                                                  mode='clean', filter_dims=[1], **{'tol':1e-5})

    # check that fundamental period in 2d dft fit is correctly assigned.
    mdl_dft, res_dft, info_dft = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[fr_len],[bl_len]], suppression_factors=[[0.],[0.]],
                                             mode='dft_leastsq', filter_dims=[1, 0], **dft_options2_2d)

    assert np.isclose(info_dft['filter_params']['axis_1']['basis_options']['fundamental_period'],
                      dft_options2_2d['fundamental_period'][1])
    assert np.isclose(info_dft['filter_params']['axis_0']['basis_options']['fundamental_period'],
                      dft_options2_2d['fundamental_period'][0])

def test_vis_clean():
    # validate that fourier_filter in various clean modes gives close values to vis_clean with equivalent parameters!
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"), bls=[(24, 25)])
    freqs = uvd.freq_array.squeeze()
    times = np.unique(uvd.time_array) * 24 * 3600
    times -= np.mean(times)
    sdf = np.median(np.diff(freqs))
    dt = np.median(np.diff(times))
    frs = np.fft.fftfreq(uvd.Ntimes, d=dt)
    dlys = np.fft.fftfreq(uvd.Nfreqs, d=sdf) * 1e9

    # simulate some data in fringe-rate and delay space
    np.random.seed(0)
    dfr, ddly = frs[1] - frs[0], dlys[1] - dlys[0]
    d = 200 * np.exp(-2j*np.pi*times[:, None]*(frs[2]+dfr/4) - 2j*np.pi*freqs[None, :]*(dlys[2]+ddly/4)/1e9)
    d += 50 * np.exp(-2j*np.pi*times[:, None]*(frs[20]) - 2j*np.pi*freqs[None, :]*(dlys[20])/1e9)
    d += 10 * ((np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes).astype(np.complex) \
         + 1j * np.random.normal(0, 1, uvd.Nfreqs * uvd.Ntimes)).reshape(uvd.Ntimes, uvd.Nfreqs))
    f = np.zeros_like(d, dtype=np.bool)
    d[:, 20:22] += 1e3
    f[:, 20:22] = True
    d[20, :] += 1e3
    f[20, :] = True
    w = (~f).astype(np.float)
    bl_len = 70.0 / 2.99e8
    # here is a fourier filter implementation of clean
    mdl1, res1, info1 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len],
                                             mode='clean', **{'tol':1e-4})

    # here is the vis filter implementation of clean
    # delay filter basic execution
    mdl2, res2, info2 = dspec.wedge_filter(d, w, bl_len, sdf, standoff=0, horizon=1.0, min_dly=0.0,
                                             tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    # validate models and residuals are close.
    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))

    mdl2, res2, info2 = dspec.vis_filter(d, w, bl_len=bl_len, sdf=sdf, standoff=0., horizon=1.0,
                                         min_dly=0.0, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)

    # cover tuple arguments.
    mdl3, res3, info3 = dspec.vis_clean(d, w, filter_size=bl_len, real_delta=sdf, tol=1e-4, window='none', skip_wgt=0.1, gain=0.1)
    # validate models and residuals are close.
    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))
    assert np.all(np.isclose(res1, res3))
    assert np.all(np.isclose(mdl1, mdl3))


    # Do the same comparison with more complicated windowing and edge cuts.
    mdl1, res1, info1 = dspec.fourier_filter(freqs, d, w, [0.], [bl_len],
                                             mode='clean', **{'tol':1e-4,
                                             'window':'tukey', 'edgecut_low':4, 'edgecut_hi':4})
    mdl2, res2, info2 = dspec.delay_filter(d, w, bl_len, sdf,
                                           edgecut_hi=4, edgecut_low=4, tol=1e-4,
                                           skip_wgt=0.1, gain=0.1, window='tukey')
    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))


    #Do a comparison for time domain clean.
    mdl1, res1, info1 = dspec.fourier_filter(times-np.mean(times), d, w, [0.], [frs[15]],
                                             mode='clean', filter_dims=[0], **{'tol':1e-4,
                                             'window':'tukey', 'edgecut_low':3, 'edgecut_hi':4})
    mdl2, res2, info2 = dspec.fringe_filter(d, w, frs[15], dt, edgecut_hi=4, edgecut_low=3,
                                            tol=1e-4, window='tukey', skip_wgt=0.1,
                                            gain=0.1)

    # test vis_clean time axis with fringe-rate clean
    mdl3, res3, info3 = dspec.vis_clean(d.T, w.T, (frs[15], frs[15]), dt, edgecut_hi=4, edgecut_low=3,
                                            tol=1e-4, window='tukey', skip_wgt=0.1,
                                            gain=0.1)
    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))
    assert np.all(np.isclose(res1, res3.T))
    assert np.all(np.isclose(mdl1, mdl3.T))

    #cover value error if 2-tuple filter sizes and not 2dclean.
    pytest.raises(ValueError, dspec.fringe_filter,d , w, frs[15], dt, edgecut_hi=4, edgecut_low=3,
                                            tol=1e-4, window='tukey', skip_wgt=0.1,
                                            gain=0.1, clean2d=True)
    #try 2d iterative clean.
    mdl1, res1, info1 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[frs[15]],[bl_len]],
                                             mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'rect','tol':1e-5,
                                             'window':['tukey', 'tukey'], 'add_clean_residual':False})

    mdl2, res2, info2 = dspec.high_pass_fourier_filter(data=d, wgts=w, filter_size=[frs[15], bl_len],
                                             real_delta=[np.mean(np.diff(times)), np.mean(np.diff(freqs))],
                                             window='tukey', tol=1e-5, clean2d=True,
                                             add_clean_residual=False)

    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))

    #check plus mode.
    mdl1, res1, info1 = dspec.fourier_filter(x=[times, freqs], data=d, wgts=w, filter_centers=[[0.],[0.]],
                                             filter_half_widths=[[frs[15]],[bl_len]],
                                             mode='clean', filter_dims=[1, 0], **{'filt2d_mode':'plus','tol':1e-5,
                                             'window':['tukey', 'tukey'], 'add_clean_residual':False})

    mdl2, res2, info2 = dspec.high_pass_fourier_filter(data=d, wgts=w, filter_size=[frs[15], bl_len],
                                             real_delta=[np.mean(np.diff(times)), np.mean(np.diff(freqs))],
                                             window='tukey', tol=1e-5, clean2d=True, filt2d_mode='plus',
                                             add_clean_residual=False)

    #cover tuple capabilities of vis_clean
    mdl3, res3, info3 = dspec.high_pass_fourier_filter(data=d, wgts=w, filter_size=[[frs[15] , frs[15]], bl_len],
                                             real_delta=[np.mean(np.diff(times)), np.mean(np.diff(freqs))],
                                             window='tukey', tol=1e-5, clean2d=True, filt2d_mode='plus',
                                             add_clean_residual=False)
    #cover value error for calling 2d visclean with 1d real_delta.
    pytest.raises(ValueError, dspec.high_pass_fourier_filter, data=d, wgts=w, filter_size=[[frs[15] , frs[15]], bl_len],
                     real_delta=np.mean(np.diff(times)),
                     window='tukey', tol=1e-5, clean2d=True, filt2d_mode='plus',
                     add_clean_residual=False)

    assert np.all(np.isclose(mdl1, mdl2))
    assert np.all(np.isclose(res1, res2))
    assert np.all(np.isclose(res3, res1))
    assert np.all(np.isclose(mdl3, mdl1))

def test_place_data_on_uniform_grid_data_already_gridded():
    # first, generate uniformly spaced x values and ensure that we get the same thing back.
    xt = np.arange(0, 100) * 1.23157
    yt = np.random.randn(len(xt)) + 1j * np.random.randn(len(xt))
    wt = np.ones(len(xt))
    wt[np.random.randint(low=0, high=len(xt), size=20)] = 0.0
    xout, yout, wout, inserted = dspec.place_data_on_uniform_grid(xt, yt, wt)
    assert np.allclose(xout, xt)
    assert np.allclose(yout, yt)
    assert np.allclose(wout, wt)

@pytest.mark.parametrize("dx", [1.23157, -3.8271])
def test_place_data_on_uniform_grid_data_on_incomplete_grid(dx):
    # test when data is not uniformly spaced but the non-uniform spacings
    # have a lowest common multiple (they are incompletely sampling a grid).
    xt = np.arange(0, 100) * dx
    yt = np.random.randn(len(xt)) + 1j * np.random.randn(len(xt))
    wt = np.ones(len(xt))
    # remove 10 random grid points and check that
    # we succisfully reconstruct grid.
    to_remove = np.random.randint(low=1, high=len(xt)-1, size=10)
    to_keep = np.array([i for i in range(len(xt)) if i not in to_remove])
    xtt = xt[to_keep]
    ytt = yt[to_keep]
    wtt = wt[to_keep]
    xout, yout, wout, inserted = dspec.place_data_on_uniform_grid(xtt, ytt, wtt)
    assert np.allclose(xout, xt)
    assert np.allclose(yout[to_keep], yt[to_keep])
    assert np.allclose(yout[to_remove], 0.0)
    assert np.allclose(wout[to_remove], 0.0)
    assert np.allclose(wout[to_keep], wt[to_keep])

def test_place_data_on_uniform_grid_not_griddable():
    # When data cannot be placed on a uniform grid,
    # just pass it through with a warning.
    x = np.array([0, 0.1, 0.25, 0.4, 0.5])
    dng = np.random.randn(len(x))
    wng = np.ones_like(x)
    with pytest.warns(RuntimeWarning):
        xout, dout, wout, inserted = dspec.place_data_on_uniform_grid(x, dng, wng)
    assert np.allclose(xout, x)
    assert np.allclose(dout, dng)
    assert np.allclose(wout, wng)
    assert np.allclose(inserted, 0.0)

def test__fit_basis_1d():
    #perform dpss interpolation, leastsq
    fs = np.arange(-50,50)
    #here is some data
    data = np.exp(2j * np.pi * 3.5/50. * fs) + 5*np.exp(2j * np.pi * 2.1/50. * fs)
    data += np.exp(-2j * np.pi * 0.7/50. * fs) + 5*np.exp(2j * np.pi * -1.36/50. * fs)
    #here are some weights with flags
    wgts = np.ones_like(data)
    wgts[6] = 0
    wgts[17] = 0
    dw = data*wgts
    dpss_opts={'eigenval_cutoff':[1e-12]}
    #perform dpss interpolation, leastsq and matrix and compare results
    mod1, resid1, info1 = dspec._fit_basis_1d(fs, dw, wgts, [0.], [5./50.], basis_options=dpss_opts,
                                    method='leastsq', basis='dpss')
    mod2, resid2, info2 = dspec._fit_basis_1d(fs, dw, wgts, [0.], [5./50.], basis_options=dpss_opts,
                                    method='matrix', basis='dpss')
    assert np.all(np.isclose(mod1, mod2, atol=1e-6))
    #perform dft interpolation, leastsq and matrix and compare results
    dft_opts={'fundamental_period':200.}

    mod4, resid4, info4 = dspec._fit_basis_1d(fs, dw, wgts, [0.], [5./50.], basis_options=dft_opts,
                                    method='leastsq', basis='dft')
    dft_opts={'fundamental_period':140.}
    #if I use 200, i get a poorly conditioned fitting matrix but fp=140 doesn't filter very well
    #not sure why this is! -AEW. leastsq does fine though, but we can't get numerically stable
    #filtering matrices for pspec analysis :(
    mod3, resid3, info3 = dspec._fit_basis_1d(fs, dw, wgts, [0.], [5./50.], basis_options=dft_opts,
                                    method='matrix', basis='dft')
    #DFT interpolation with leastsq does OK.
    #DFT interpolation with matrix method is problematic.
    #Matrices with 2B harmonics are poorly conditioned which is
    #unfortunate since 2B is generally where DFT performance
    #approaches DPSS performance.
    assert np.all(np.isclose(mod4, mod2, atol=1e-5))
    assert np.all(np.isclose(mod3, mod4, atol=1e-2))

    assert np.all(np.isclose((mod2+resid2)*wgts, dw, atol=1e-6))

def test_fit_basis_1d_with_missing_channels():
    fs = np.arange(-50,50)
    #here is some data
    data = np.exp(2j * np.pi * 3.5/50. * fs) + 5*np.exp(2j * np.pi * 2.1/50. * fs)
    data += np.exp(-2j * np.pi * 0.7/50. * fs) + 5*np.exp(2j * np.pi * -1.36/50. * fs)
    #here are some weights with flags
    wgts = np.ones_like(data)
    wgts[6] = 0
    wgts[17] = 0
    dw = data*wgts
    # dft fitting options
    dpss_opts={'eigenval_cutoff':[1e-12, 1e-12]}
    # now remove ten random channels.
    to_remove = [2, 10, 11, 23, 54, 71, 87, 88, 89, 97]
    to_keep = np.array([i for i in range(len(fs)) if i not in to_remove])

    mod5, resid5, info5 = dspec._fit_basis_1d(fs[to_keep], dw[to_keep], wgts[to_keep], [0., 10/50.], [5./50., 1./50.], basis_options=dpss_opts,
                                    method='leastsq', basis='dpss')

    dwt = copy.deepcopy(dw)
    wgtst = copy.deepcopy(wgts)

    dwt[to_remove] = 0.0
    wgtst[to_remove] = 0.0

    mod6, resid6, info6 = dspec._fit_basis_1d(fs, dwt, wgtst, [0., 10/50.], [5./50., 1./50.], basis_options=dpss_opts,
                                    method='leastsq', basis='dpss')

    assert np.allclose(mod5, mod6[to_keep])
    assert np.allclose(resid5, resid6[to_keep])

def test_fit_basis_2d_nanwarning():
    # test that info propagates skips when there is an SVD convergence error.
    fs = np.arange(-50,50)
    ts = np.arange(93)
    #here is some data
    data = np.exp(2j * np.pi * 3.5/50. * fs) + 5*np.exp(2j * np.pi * 2.1/50. * fs)
    data += np.exp(-2j * np.pi * 0.7/50. * fs) + 5*np.exp(2j * np.pi * -1.36/50. * fs)
    data = np.asarray([data for i in range(93)])
    #here are some weights with flags
    wgts = np.ones_like(data)
    data[68, 12] = np.inf
    # dft fitting options
    dpss_opts={'eigenval_cutoff':[1e-12, 1e-12]}
    with pytest.warns(RuntimeWarning):
        model, resid, info = dspec._fit_basis_2d(fs, data, wgts, [0., 10/50.], [5./50., 1./50.], basis_options=dpss_opts,
                                                 method='leastsq', basis='dpss', filter_dims=1)
        assert list(np.where([info['status']['axis_1'][i] == 'skipped' for i in range(93)])[0]) == [68]

    with pytest.warns(RuntimeWarning):
        model, resid, info = dspec._fit_basis_2d(ts, data, wgts, [0., 10/50.], [5./50., 1./50.], basis_options=dpss_opts,
                                                 method='leastsq', basis='dpss', filter_dims=0)
        assert list(np.where([info['status']['axis_0'][i] == 'skipped' for i in range(100)])[0]) == [12]
