###############################################################################
# model_flux.pyx
###############################################################################
#
# Calculate the total predicted DM + HESS bkg model fluxt for a given set
# of model parameters
#
# For the endpoint or endpoint + continuum case these spectra should be
# precalculated to speed up the calculations, as envision this function
# being called directly by the likelihood
#
# Here we precalculate the smoothing for speed and in cython
#
###############################################################################


# Import basic functions
import numpy as np
cimport numpy as np
import numba
print(numba.__version__)
cimport cython
import array as arr
from itertools import product
from time import time
from scipy import integrate
from numba import complex128,float64,jit
from multiprocessing import Pool
# C functions
cdef extern from "math.h":
    double pow(double x, double y) nogil
    double sqrt(double x) nogil
    double log(double x) nogil
    double log10(double x) nogil
    double exp(double x) nogil

# Useful variables
cdef double pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
#cpdef flux(double l10xsec, double[::1] binspec, double l10a0, double a1, 
   #        double a2, double a3, double beta, double mux, double sigmax):
    # Combined model flux E^2.7 dN/dE, in units of
    #        [ TeV^1.7 m^-2 s^-1 sr^-1]
     #               units are chosen following Fig. 1 of 1301.1173
      #                      l10xsec: log10 gamma gamma + gamma Z/2 in [cm^3/s]
       #                             binspec: precomputed smoothed DM spec
        #                                    l10a0,a1,a2,a3,beta,mux,sigmax: HESS bkg model params """

   # cdef double[::1] model_flux = np.zeros(48)
    
   # cdef Py_ssize_t i

   # for i in range(48):
        # Add background
       # for j in range(37):
                
            # model_flux[i] += bkg_spec(E, l10a0, a1, a2, a3, beta, mux, sigmax, omega[j])
        # Add signal
            # model_flux[i] += binspec[i]*pow(10.,l10xsec)
        # Account for E^2.7
            # model_flux[i] *= pow(Hebins[i],2.7)
    

# return model_flu
#####################
#Number of ON events#
#####################
#from collections import OrderedDict
#old_settings = np.seterr(all='print')
#OrderedDict(np.geterr())

cpdef numberevents_ON(spec_arr):
  #  start = time()    
    ON_value = 1 
    load = np.loadtxt('../data/BinnedSpectra_Matt.dat')
    spec_arr = load[:,2:]
    scalar = (1/JTotal)*(Hebins[47]-Hebins[0])*((10**9)*(10**-4))*(403200)
    
   # for k in range(8):
      #  for n in range(301):
       #     for i in range(79):
            #ON_value *= (JHess[k]/JTotal)*integrate.quad(f4,Hebins[1],Hebins[48],args=(spec_arr[i,n]))[0] # J fraction is the second dimention.
   # for i, n in product(range(79), range(301)):

    ON_value = np.prod([JHess[z]*spec_arr*scalar for z in range(8)])

 #   print('time: {}s'.format(time() - start))
    return ON_value
##########################################
#integration function number of ON events#
##########################################
#cpdef f4(double E, spec_arr):
 #   f4_value = 1
  #  load = np.loadtxt('../data/BinnedSpectra_Matt.dat')
#    spec_arr = load[:,2:]
 #   for n in range(301):
  #      for i in range(79):
   #         f4_value *= (10^9*10^-4)*(403200)*spec_arr[i,n]
    #        return f4_value 
       # return (10^9*10^-4)*(403200)*spec_arr[:,i]  # multiply effective area with e-4 from cm2 to m2, observation time, spectra 

#########################
#Number of Signal Events#
#########################
 
cpdef numberevents_signalspec(double E,int j,double mux, binspec, double l10xsec):
    x1 = Hebins[j]-E 
    G1 = exp(-pow(x1 - mux,2.)/(2.*pow(Hebins[j]/10,2.))) / sqrt(2.*pi*pow(Hebins[j]/10,2.))
    f1 = binspec*pow(10.,l10xsec)*G1
    signal = 403200*integrate.nquad(f1,[[-np.inf, np.inf],[Hebins[j]-(Hebins[j+1]-Hebins[j])/2,Hebins[j]+(Hebins[j+1]-Hebins[j])/2]])[0]
    
    return signal 
#args=(j,mux, binspec,l10xsec),
#cpdef f1(double E,int j,double mux, binspec,double l10xsec):
                                                                              
 #   signal = integrate.quad(count_rates_signalspecfull_output=1)[0]
   
  #  return
#cpdef count_rates_signalspec(int j,double mux, binspec, double l10xsec):
 #   return integrate.quad(f1,-np.inf,+np.inf,args=(j,mux,binspec,l10xsec),full_output=1)[0] 
  #  return signal*403200
   # return integrate.nquad(f3,Hebins[j]-(Hebins[j+1]-Hebins[j])/2,Hebins[j]+(Hebins[j+1]-Hebins[j])/2,args=(j, mux, binspec,l10xsec),full_output=1)[0]
##############################################
#integration function number of signal events#
##############################################
#cpdef f3(double E,int j,double mux, binspec,double l10xsec):
#    return count_rates_signalspec(j,mux, binspec,l10xsec)*403200

#############################
#Number of Background Events#
#############################
cpdef numberevents_bkgspec(double E, double l10a0,double a1,double a2,double a3,double beta,double mux,double sigmax,int j,int k):
    x = log10(E)                                                                
    P = exp(a1*x + a2*pow(x,2.) + a3*pow(x,3.)) 
    G = exp(-pow(x - mux,2.)/(2.*pow(sigmax,2.))) / sqrt(2.*pi*pow(sigmax,2.))
    bkg_spec = pow(10.,l10a0)*pow(E,-2.7)*(P + beta*G)*omega[k]
    x1 = Hebins[j]-E 
    G1 = exp(-(pow(x1 - mux,2.)/(2.*pow(Hebins[j]/10,2.)))) / sqrt(2.*pi*pow(Hebins[j]/10,2.)) 
    integrant1 = bkg_spec*G1*0.1
    return 403200*integrate.nquad(bkg_spec,[[np.inf,-np.inf],[Hebins[j]-(Hebins[j+1]-Hebins[j])/2,Hebins[j]+(Hebins[j+1]-Hebins[j])/2]])
         #   args=(l10a0,a1,a2,a3,beta,mux,sigmax,k,j),full_output=1)[0] 
# bkg_spec(E, l10a0, a1, a2, a3,beta, mux,sigmax,k)*G1*0.1
   #print(l10a0, a1, a2, a3, beta, mux, sigmax,j,k)
   # print(Hebins[j]-(Hebins[j+1]-Hebins[j])/2)
   # print(Hebins[j]+(Hebins[j+1]-Hebins[j])/2)
#   return integrate.quad(f2args=(l10a0, a1, a2, a3, beta, mux, sigmax,j,k),full_output=1)[0]
##################################################
#integration function number of background events#
##################################################
#cpdef f2(double E, double l10a0,double a1,double a2,double a3,double beta,double mux,double sigmax,int j,int k):
 #   return count_rates_bkgspec(l10a0, a1, a2, a3, beta, mux, sigmax,j,k)*403200

########################
#Signal Spec Count Rate#
########################
#from scipy import integrate as integrate


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)

 # binspec: precomputed smoothed DM spec
  #integrate.quad(f1,0,np.inf,args=(j,mux,binspec,i,l10xsec),full_output=1)[0]
#############################################
#integral function of Signal Spec Count Rate#
#############################################



   



##############################
# Background Spec Count Rate #
##############################
#from scipy import integrate as integrate


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)

#cdef count_rates_bkgspec(double l10a0, double a1, double a2, double a3,
#               double beta, double mux, double sigmax,int k,int j):
    
 #   return #integrate.quad(f,0,np.inf,args=(l10a0,a1,a2,a3,beta,mux,sigmax,k,j))[0]
 ###################################################
 # integral function of Background Spec Count Rate #
 ###################################################
#cdef f(double E,double l10a0, double a1, double a2, double a3,
#                                             double beta, double mux, double sigmax,int k,int j):
 
 
   
    
####################
# Background Model #
####################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)


#cdef double bkg_spec(double E, double l10a0, double a1, double a2, double a3, 
 #           double beta, double mux, double sigmax, int k):
#    """ The 7 parameter background model as given in Eq. 1 of 1301.1173
 #       E is an array of energies in TeV, and the remainder are the 7 params

  #      Returns dN/dE(E) in whatever units l10a0 carries
   # """

   
   
   
    
   


###################
# HESS Parameters #
###################

# Define the HESS data point energies (where we want the model flux)
# energy values of points from Fig. 1 of 1301.1173 - in TeV
def Hebins(j):
    Hebins = [0.31216343, 0.34619903, 0.37656738, 0.41493257, 0.45425882,
          0.50378727, 0.54797916, 0.59991523, 0.65677367, 0.719021,
          0.78716797, 0.86736571, 0.95573409, 1.03957047, 1.15291623,
          1.25404929, 1.38181352, 1.51277828, 1.65615555, 1.81312174,
          1.99784503, 2.18719569, 2.39449251, 2.6214364, 2.86988946,
          3.16227766, 3.48445477, 3.79010887, 4.17625027, 4.54258822,
          5.00539323, 5.51534944, 6.03808019, 6.65324794, 7.23686651,
          7.97416822, 8.67365706, 9.55734093, 10.46316132, 11.5291623,
          12.62186688, 13.81813522, 15.22594546, 16.66902172, 18.36728411,
          20.10808843, 22.01388175, 24.10030129]
cdef double omega(k):
    omega = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# J-factor appropriate for the HESS analysis
# As explained in 1103.3266 their ROI is defined by r<1 deg and |b|>0.3
# Also they use an Einasto profile with alpha=0.17, rs=20, rho0=0.39
# Then the average J factor over the ROI in TeV^2/cm^5
cdef double JHess(z):
    JHess = [3.80113*10**20,5.21468*10**20,6.22573*10**20,6.97463*10**20,7.54186*10**20,7.97764*10**20,8.31518*10**20,8.57745*10**20]
JTotal = 5.46283*10**21
