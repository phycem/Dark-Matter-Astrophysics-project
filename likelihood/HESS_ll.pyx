###############################################################################
# HESS_ll.pyx
###############################################################################
#
# Calculate the variable part of the log likelihood for a fit to the digitized
# HESS data. We model this as a Gaussian likelihood, but as for sensitivity
# studies we only care about differences in ll, we throw out the normalizing
# prefactor.
#
# We determine the appropriate standard deviation asymmetrically: if the model
# is above the HESS mean we use the upper error and vice versa.
#
# Here we precalculate the smoothing for speed and its in cython
#
###############################################################################

# Import basic functions
import numpy as np
cimport numpy as np
cimport cython
import model_flux as mf
from math import factorial
from itertools import product



# C functions
cdef extern from "math.h":
    double pow(double x, double y) nogil
    double exp(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double ll(double l10xsec, binspec, double l10a0, double a1, 
               double a2, double a3, double beta, double mux, double sigmax):
    """ Calculate the log likelihood as a function of
        l10xsec: log10 gamma gamma + gamma Z/2 in [cm^3/s]
        binspec: precomputed smoothed DM spec
        l10a0,a1,a2,a3,beta,mux,sigmax: HESS bkg model params
    """

    # Get model flux in units of [ TeV^1.7 m^-2 s^-1 sr^-1]
    # Units same as HESS data in Fig. 1 of 1301.1173

    # Loop through values, if below use lower error, if above use upper error
    cdef Py_ssize_t k
    cdef Py_ssize_t j    
    cdef double l = 1.0
    load = np.loadtxt('../data/BinnedSpectra_Matt.dat')
    spec_arr = load[:,2:]
    cdef double nson = 1.0
    cdef double nb = 1.0
    cdef double non = 1.0
   # for j in range(48):        
      #  for k in range(8):
    for j in range(48):
        nson = mf.numberevents_signalspec(j,mux,binspec,l10xsec)
#        nb = mf.numberevents_bkgspec(l10a0,a1,a2,a3,beta,mux,sigmax,j,k)
        non = mf.numberevents_ON(spec_arr)
        l = l * exp(-1*(nson - nb)) * ((nson + nb)**non) * (1/factorial(np.nan_to_num(non))) 
           # with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.log(l)
    """
    for i in range(48):
        if model_flux[i] < data[i]:
            ll -= pow(model_flux[i]-data[i],2.) / (2.*pow(sigl[i],2.))
        else:
            ll -= pow(model_flux[i]-data[i],2.) / (2.*pow(sigu[i],2.))

    return ll
    """


#############
# HESS data #
#############

# Digitized central, lower, and upper errors of HESS data in Fig. 1 of 1301.1173
# These are the digitized log10 values

cdef double Hebins_c[48]
cdef double Hebins_l[48]
cdef double Hebins_u[48]

HESS_c = [-3.32454361, -3.35699797, -3.35699797, -3.36916836, -3.43002028,
          -3.45030426, -3.52738337, -3.55172414, -3.60446247, -3.61257606,
          -3.62068966, -3.64503043, -3.65720081, -3.7505071, -3.67748479,
          -3.76267748, -3.72616633, -3.75456389, -3.78296146, -3.80324544,
          -3.77890467, -3.77079108, -3.79107505, -3.7464503, -3.76673428,
          -3.67748479, -3.71399594, -3.71805274, -3.6653144, -3.62474645,
          -3.54361055, -3.57200811, -3.5801217, -3.53549696, -3.69371197,
          -3.55172414, -3.71399594, -3.49087221, -3.58823529, -3.46247465,
          -3.50709939, -3.67748479, -3.39350913, -3.43813387, -3.79107505,
          -3.53955375, -3.63691684, -4.15212982]

HESS_l = [-3.34077079, -3.36916836, -3.37322515, -3.38539554, -3.45030426,
          -3.47058824, -3.55578093, -3.57606491, -3.62677485, -3.63286004,
          -3.64097363, -3.6653144, -3.68356998, -3.78498986, -3.70791075,
          -3.79918864, -3.76064909, -3.79107505, -3.82555781, -3.85192698,
          -3.82758621, -3.81135903, -3.84381339, -3.80324544, -3.82150101,
          -3.73630832, -3.77281947, -3.78296146, -3.72616633, -3.69371197,
          -3.60649087, -3.64705882, -3.65314402, -3.61460446, -3.80121704,
          -3.64908722, -3.84584178, -3.59634888, -3.71196755, -3.5841785,
          -3.64908722, -3.86815416, -3.53752535, -3.61257606, -4.0872211,
          -3.77079108, -3.93306288, -4.9959432]

HESS_u = [-3.30425963, -3.34077079, -3.34077079, -3.36105477, -3.4137931,
          -3.43407708, -3.50709939, -3.53144016, -3.5841785, -3.59634888,
          -3.59229209, -3.62474645, -3.62880325, -3.72210953, -3.65720081,
          -3.73427992, -3.69776876, -3.72616633, -3.7505071, -3.77079108,
          -3.73833671, -3.71805274, -3.7464503, -3.71399594, -3.72616633,
          -3.63691684, -3.66125761, -3.66125761, -3.60851927, -3.56795132,
          -3.48275862, -3.51926978, -3.51115619, -3.46247465, -3.60851927,
          -3.47464503, -3.61663286, -3.40567951, -3.48275862, -3.36916836,
          -3.39756592, -3.55172414, -3.29208925, -3.32048682, -3.60851927,
          -3.39756592, -3.45841785, -3.85598377]

# Convert to actual data and error bars
cdef Py_ssize_t iE
cdef double data[48]
cdef double sigl[48]
cdef double sigu[48]

for iE in range(48):
    data[iE] = pow(10.,HESS_c[iE])
    sigl[iE] = data[iE] - pow(10.,HESS_l[iE])
    sigu[iE] = pow(10.,HESS_u[iE]) - data[iE]
