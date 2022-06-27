###############################################################################
# clean_LL.pyx
###############################################################################
#
# clean LLs by trying to remove the spikes minuit injects due to instability
#
###############################################################################

import numpy as np
from cython.parallel import parallel, prange
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double sqrt(double x) nogil
    double fabs(double x) nogil

cdef double pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double[::1] clean_LL(double[::1] LL,double thresh=0.0001):
    cdef int Nll = len(LL)
    cdef double[::1] LL_new = np.zeros(Nll,dtype = DTYPE)
    #return LL_new
    
    cdef int down = 0
    cdef double zero = LL[0]
    cdef double top = find_max(LL[0:20],20)

    cdef double[::1] LL_2 = np.zeros(Nll,dtype=DTYPE)
    cdef Py_ssize_t k
    with nogil:
        for k in range(Nll):
            if k < 20:
                LL_2[k] = top
            else:
                LL_2[k] = LL[k]
    #LL[0:20] = np.ones(20,dtype = DTYPE)*top
    cdef double LL_m1 = LL_2[0]
        
    cdef double cur_val = 0.0
    cdef double next_val = 0.0
    cdef int t1 = 0
    cdef int t2 = 0
    cdef int t3 = 0
    cdef Py_ssize_t i

    with nogil:
        for i in range(0,Nll-1):
            if LL_2[i] > zero*(1+thresh):
                down= 0
            cur_val = (LL_2[i] - LL_m1)/fabs(LL_2[i] + LL_m1)
            next_val = (LL_2[i+1] - LL_2[i])/fabs(LL_2[i+1] + LL_2[i])

            t1 = fabs(cur_val) > thresh 
            t2 = cur_val < thresh
            t3 = down
            if (t1 and t2) or t3:
                if fabs(next_val) > thresh and next_val < thresh:
                    LL_new[i] = LL_2[i]
                    LL_m1 = LL_2[i]
                    down = 0
                else:
                    LL_new[i] = LL_m1
                    down = 1
            else:
                LL_new[i] = LL_2[i]
                LL_m1 = LL_2[i]
        LL_new[Nll-1] = LL_2[Nll-1]
    return LL_new

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double find_max(double[::1] x, int Nx) nogil:
    """ Manually find maximum
    """
    cdef Py_ssize_t i
    cdef double res = x[0]
    for i in range(1,Nx):
        if x[i] > res:
            res = x[i]
    return res
