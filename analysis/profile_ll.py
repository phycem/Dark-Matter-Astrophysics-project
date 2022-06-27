###############################################################################
# profile_ll.py
###############################################################################
#
# Find the limit as a function of mass by profiling over background parameters
#
# Here we precalculate the smoothing for speed and its in cython
#
# Now use more masses and Matt's Sommerfeld parameters
#
###############################################################################

# Basic modules
import numpy as np
import sys
import copy
from tqdm import *

# Load custom functions
sys.path.append('../1/')
import HESS_ll as hl
import preload_spec as ps
import clean_LL as cl

# Minuit and related functions
from iminuit import Minuit
from minuit_functions import call_ll

class scan_ll():
    def __init__(self, stype, verbose=0):

        self.verbose = verbose # whether Minuit prints output
        self.find_lim(stype)


    def find_lim(self, stype):
        """ For a given stype compute and save the limits for an array of masses
        """
        
        # Load mass values
        self.mass_arr()

        lim_ary = np.zeros((len(self.mvals),2))

        for mi in tqdm(range(len(self.mvals))):
            lim_ary[mi,0] = self.mvals[mi]
            lim_ary[mi,1] = self.find_m_lim(self.mvals[mi], stype)

        np.savetxt('./output/lim_ar_'+str(stype)+'.dat', lim_ary)


    def find_m_lim(self, mass, stype):
        """ For a given mass and stype find the xsec limit
        """

        # Setup preloaded spectrum
        self.binspec = ps.sig_spec(mass, stype)
        
        # Calculate the LLs as a function of xsec
        xsecs = np.logspace(-31,-24,1001)
        LL_arr = np.zeros(1001)
        for xi in tqdm(range(1001)):
            LL_arr[xi] = self.run(np.log10(xsecs[xi]))

        # Double clean LLs from minuit instabilities
        LL_arr = cl.clean_LL(LL_arr, thresh=1e-8)
        LL_arr = cl.clean_LL(LL_arr, thresh=1e-8)
        LL_arr = np.array(LL_arr)

        # Determine the TS, and find where 2.71 from the max
        TS_xsec_ary = 2*(LL_arr-LL_arr[0])
        max_loc = np.argmax(TS_xsec_ary)
        max_TS = TS_xsec_ary[max_loc]

        #for xi in range(max_loc,len(xsecs)):
        #    val = TS_xsec_ary[xi] - max_TS
        for xi in range(len(xsecs)):
            val = TS_xsec_ary[xi]
            if val < -2.71:
                scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
                lim = 10**(np.log10(xsecs[xi-1])+scale*(np.log10(xsecs[xi])-np.log10(xsecs[xi-1])))
                break

        return lim


    def run(self, logxsec):
        """ For given DM parameters run and return the LL
        """

        self.logxsec = logxsec

        # Setup keys
        keys = np.array(['l10a0','a1','a2','a3','beta','mux','sigmax'])
        limit_dict = {}
        init_val_dict = {}
        step_size_dict = {}

        limit_dict['limit_l10a0'] = (-10,-1)
        limit_dict['limit_a1'] = (-3,3)
        limit_dict['limit_a2'] = (-3,3)
        limit_dict['limit_a3'] = (-3,3)
        limit_dict['limit_beta'] = (0,10)
        limit_dict['limit_mux'] = (-2,2)
        limit_dict['limit_sigmax'] = (0.0001,100)

        init_val_dict['l10a0'] = -4. 
        init_val_dict['a1'] = 0.
        init_val_dict['a2'] = 0.
        init_val_dict['a3'] = 0.
        init_val_dict['beta'] = 1.
        init_val_dict['mux'] = 0.
        init_val_dict['sigmax'] = 1.
        

        for key in keys:
            step_size_dict['error_'+key] = 1

        other_kwargs = {'print_level': self.verbose, 'errordef': 1}

        # Parse into form Minuit wants
        z = limit_dict.copy()
        z.update(other_kwargs)
        z.update(limit_dict)
        z.update(init_val_dict)
        z.update(step_size_dict)
        fm = call_ll(len(keys),self.ll,keys)
        m = Minuit(fm,**z)

        # Run and print out best fit values
        m.migrad(ncall=20000, precision=1e-14)

        return self.ll([m.values['l10a0'],m.values['a1'],m.values['a2'],
                        m.values['a3'],m.values['beta'],m.values['mux'],
                        m.values['sigmax']])


    def ll(self, theta, ndim=1, nparams=1):
        """ Define the ll from HESS_ll - DM params set through self
        """

        return hl.ll(self.logxsec, self.binspec, theta[0], theta[1], theta[2], 
                     theta[3], theta[4], theta[5], theta[6])

    def mass_arr(self):
        """ Define the masses to scan over [TeV]
        """

        #self.mvals = 0.25*(2.+np.arange(79))
        self.mvals = np.logspace(-0.301029995664,1.30102999566,200)
