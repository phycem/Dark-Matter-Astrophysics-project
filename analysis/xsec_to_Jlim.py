###############################################################################
# xsec_to_Jlim.py
###############################################################################
#
# Convert a limit on xsec to one on J assuming the standard gamma gamma +
# gamma Z/2 cross section
#
###############################################################################

import numpy as np

def convert(stype, sym=0):
    """ Convert xsec to J limit, for a given stype and sym or asym errors
    """

    if sym:
        xlim = np.loadtxt('./output/sym_lim_ar_'+str(stype)+'.dat')
    else:
        xlim = np.loadtxt('./output/lim_ar_'+str(stype)+'.dat')

    # J-factor appropriate for the HESS analysis
    # As explained in 1103.3266 their ROI is defined by r<1 deg and |b|>0.3
    # Also they use an Einasto profile with alpha=0.17, rs=20, rho0=0.39
    # Then the average J factor over the ROI in TeV^2/cm^5

    JHess = 7.391585986662391e18

    # Now load the appropriate gamma gamma + gamma Z/2 xsec and rescale
    lineload = np.loadtxt('../data/LLxsec_new.dat')
    linemvals = lineload[:,0] # TeV
    
    # Convert to a J limit
    Jlim = np.zeros((len(xlim),2))
    for Ji in range(len(xlim)):
        mass = xlim[Ji,0]
        linexsec = np.interp(mass, linemvals, lineload[:,1]) # [cm^3/s]

        Jlim[Ji,0] = mass
        Jlim[Ji,1] = xlim[Ji,1] * JHess / linexsec
    
    if sym:
        np.savetxt('./output/sym_Jlim_ar_'+str(stype)+'.dat', Jlim)
    else:
        np.savetxt('./output/Jlim_ar_'+str(stype)+'.dat', Jlim)
