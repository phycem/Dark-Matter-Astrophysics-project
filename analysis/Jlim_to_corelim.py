###############################################################################
# Jlim_to_corelim.py
###############################################################################
#
# Convert a limit on the J-factor to an NFW core size
#
###############################################################################

import numpy as np

def convert(stype):
    """ Convert xsec to J limit, for a given stype
    """

    Jlim = np.loadtxt('./output/Jlim_ar_'+str(stype)+'.dat')

    # Load the core size vs J-factor array
    coreJ = np.loadtxt('../data/coreJ.dat')

    # Reverse order for interpolation
    corevals = coreJ[:,0][::-1]
    coreJvals = coreJ[:,1][::-1]

    # Convert to a J limit
    corelim = np.zeros((len(Jlim),2))
    for corei in range(len(Jlim)):
        corelim[corei,0] = Jlim[corei,0]
        corelim[corei,1] = np.interp(Jlim[corei,1],coreJvals,corevals)
        
    np.savetxt('./output/corelim_ar_'+str(stype)+'.dat', corelim)
