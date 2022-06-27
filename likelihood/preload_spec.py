###############################################################################
# preload_spec.py
###############################################################################
#
# For a given mass preload the endpoint and continuum spectra.
#
# Here we precalculate the smoothing for speed.
#
###############################################################################

import numpy as np
import array as arr
################
# Signal Model #
################

def sig_spec(mass, stype):
    """ Signal prediction minus cross section factor
    """

    # Line - after convolution becomes a Gaussian
    # Line is always present, so add this in regardless
    # Multiply by 2 for the same reason there is a 2 in the delta function
    width = HESS_eres([mass])[0]
    dNdE = 2.*np.exp(-(Hebins-mass)**2./(2.*width**2.)) \
            / np.sqrt(2.*np.pi*width**2.)

    # Line + endpoint, if continuum just add that at the end
    if stype==1 or stype==2:
        load_binspec = ep_spec(mass)
        deltaamp = load_binspec[0]
        spec = load_binspec[1:]

        spec_E = mass*(1.-10.**(-np.linspace(0,3,301)))
        spec_dE = mass*(10.**(-0.01*(-0.5+np.arange(301,dtype=np.float)))
                      -(10.**(-0.01*(0.5+np.arange(301,dtype=np.float)))))

        # Pad spectra with 0s above the mass
        spec_E = np.append(spec_E,mass*(1.+0.01*np.arange(50,dtype=np.float)))
        spec_dE = np.append(spec_dE,mass*0.01+np.zeros(50))
        spec = np.append(spec,np.zeros(50))

        # Normalize to 2*delta function, so that the xsec is
        # for gamma gamma + gamma Z/2
        norm_l = deltaamp*mass
        spec *= 2./norm_l

        # If asked for add the continuum spectrum - should be pre-normalized
        if stype==2:
            load_cspec = ct_spec(mass)
            cspec = np.append(load_cspec,np.zeros(50))
            spec += cspec

        # Cut the data below 100 GeV as HESS effective area drops steeply
        sub_100 = np.where(spec_E < 0.1)[0]
        spec[sub_100] = 0.

        # Convolve this with the energy resolution
        width = HESS_eres(spec_E)
        for i in range(len(spec_E)):

            if width[i] != 0.: # skip padding
                dNdE += spec_dE[i] * spec[i] \
                * np.exp(-(Hebins - spec_E[i])**2./(2.*width[i]**2)) \
                / np.sqrt(2.*np.pi*width[i]**2)

    # Norm in m^-2 s^-1 TeV^-1 sr^-1 - 1.e4 converts cm^-2 to m^-2
    for k in range(8):
        norm = JHess[k] / (8.*np.pi*mass**2.) * 1.e4
    return norm
    fluxnoxsec = norm*dNdE

    return fluxnoxsec


#####################
# Endpoint Spectrum #
#####################

def ep_spec(mass):
    """ Load the endpoint dN/dE spectrum for a given mass [TeV]
    """

    # Load the precomputed data and extract various elements
    load = np.loadtxt('../data/BinnedSpectra_Matt.dat')
    mass_arr = load[:,0]*1.e-3 # convert from GeV to TeV
    line_amp_arr = load[:,1]
    spec_arr = load[:,2:]

    # Interpolate to the relevant mass
    ints = np.arange(79,dtype=np.float)
    mass_int = np.interp(mass, mass_arr, ints)
    
    line_amp_val = np.interp(mass_int, ints, line_amp_arr)
    spec_vals = np.array([np.interp(mass_int, ints, spec_arr[:,i]) for i in range(301)])

    # Repackage the arrays and return

    return np.append(np.array([line_amp_val]),spec_vals)


######################
# Continuum Spectrum #
######################

def ct_spec(mass, withZ=1):
    """ Load the continuum spectrum for a given mass [TeV]
        
        The spectra are based on the files from PPPC4DMID, described in
        M. Cirelli et al., JCAP 1103, 051 (2011), 1012.4515
    """

    # Load spectrum, keeping track of header
    with open('../data/AtProductionNoEW_gammas.dat') as f:
        lines = [line for line in f if not line.startswith('#')]
        data = np.genfromtxt(lines, names = True, dtype = None)

    mass_arr = data['mDM']*1.e-3 # convert from GeV to TeV
    mass_unq = np.unique(mass_arr)
    xvals = np.unique(10.**data['Log10x'])
    
    # Create 2d arrays mass versus xvals
    # Spectra are in dN/dlog10(x)
    W_arr = np.zeros((len(mass_unq),len(xvals)))
    Z_arr = np.zeros((len(mass_unq),len(xvals)))
    for mi in range(len(mass_unq)):
        mwhere = np.where(mass_arr == mass_unq[mi])[0]
        W_arr[mi] = data['W'][mwhere]
        Z_arr[mi] = data['Z'][mwhere]

    # Now interpolate spectra to the relevant mass
    W_spec = np.zeros(len(xvals))
    Z_spec = np.zeros(len(xvals))
    ints = np.arange(len(mass_unq),dtype=np.float)
    for xi in range(len(xvals)):
        mass_int = np.interp(mass, mass_unq, ints) 
        W_spec[xi] = np.interp(mass_int, ints, W_arr[:,xi])
        Z_spec[xi] = np.interp(mass_int, ints, Z_arr[:,xi])

    # Pad spectrum with zeros at the end
    xvals = np.append(xvals, np.array([1., 1.05, 1.1]))
    W_spec = np.append(W_spec, np.array([0., 0., 0.]))
    Z_spec = np.append(Z_spec, np.array([0., 0., 0.]))

    # Convert to physical from dimensionless values
    Evals = mass*xvals
    W_spec /= Evals*np.log(10)
    Z_spec /= Evals*np.log(10)

    # Pad with zero at the start (do now to avoid 1/0 issue)
    Evals = np.append(np.array([0.]), Evals)
    W_spec = np.append(np.array([0.]), W_spec)
    Z_spec = np.append(np.array([0.]), Z_spec)

    # Now normalize these to the gamma gamma + gamma Z/2 xsec
    full_spec = W_spec * ww_rescale(mass)

    if withZ:
        # Ratio of (ZZ + gamma Z/2) / (gamma gamma + gamma Z/2) is
        # cW^2/sW^2 (e.g. for ZZ + gamma Z/2 we have cW^4 + sW^2cW^2 = cW^2)
        full_spec += Z_spec * (cWsq/sWsq)

    # Interpolate to the appropriate energy bins 
    spec_E = mass*(1.-10.**(-np.linspace(0,3,301)))
    output_spec = np.interp(spec_E, Evals, full_spec)
    
    return output_spec


###########
# WW xsec #
###########

def ww_rescale(mass):
    """ Calculate the cross section to WW and take the ratio of this to the
        gamma gamma + gamma Z/2 xsec
    """

    # Define factors, following notation in (A8) of 1307.4082
    # Here 1=+-, 2=00
    Gamma11 = np.pi*alpha2**2./(4.*mass**2)*2.
    Gamma12 = np.pi*alpha2**2./(4.*mass**2)*2.*np.sqrt(2.)
    Gamma22 = np.pi*alpha2**2./(4.*mass**2)*4.

    # Load Sommerfeld sij values, interpolate to our mass
    sload = np.loadtxt('../data/Sommerfeld_v=1e-3.dat')
    mvals = sload[:,0]*1.e-3 # convert GeV to TeV
    s22re = np.interp(mass, mvals, sload[:,1])
    s22im = np.interp(mass, mvals, sload[:,2]) 
    s21re = np.interp(mass, mvals, sload[:,3])
    s21im = np.interp(mass, mvals, sload[:,4])

    # Compute combinations of these we need
    abss21sq = s21re**2 + s21im**2
    res21cs22 = s21re*s22re + s21im*s22im
    abss22sq = s22re**2 + s22im**2

    # Final xsec
    wxsec = 2.*(Gamma11*abss21sq + 2.*Gamma12*res21cs22 + Gamma22*abss22sq)
    wxsec *= 1.1673299710900705e-23 # convert [TeV^-2] to [cm^3/s]

    # Now load the appropriate gamma gamma + gamma Z/2 xsec and rescale
    lineload = np.loadtxt('../data/LL_line_cross_section.dat')
    linemvals = lineload[:,0] # TeV
    linexsec = np.interp(mass, linemvals, lineload[:,3]) # [cm^3/s]

    return wxsec/linexsec 


#################
# EW Parameters #
#################

# All in MSbar at the Z pole
sWsq = 0.23126 # see PDG review page 8 table 10.2
cWsq = 1. - sWsq
alphaMZ = 1./127.944 # see PDG review page 4
alpha2 = alphaMZ / sWsq


#####################
# Energy Resolution #
#####################

def HESS_eres(E):
    """ Return the Gaussian width of the HESS energy resolution for an array of
        energies E [TeV]

        From the right column of page 3 in 1301.1173, this is 17% at 0.5 TeV
        and 11% at 10 TeV. We log interpolate between these boundaries
    """

    width = np.zeros(len(E))
    for i in range(len(E)):
        if E[i] == 0.:
            width[i] = 0.
        else:
            width[i] = E[i]*np.interp(np.log(E[i]), [np.log(0.5), np.log(10.)], [0.17, 0.11])

    return width


###################
# HESS Parameters #
###################

# Define the HESS data point energies (where we want the model flux)
# energy values of points from Fig. 1 of 1301.1173 - in TeV
Hebins = np.array([0.31216343, 0.34619903, 0.37656738, 0.41493257, 0.45425882,
                    0.50378727, 0.54797916, 0.59991523, 0.65677367, 0.719021,
                    0.78716797, 0.86736571, 0.95573409, 1.03957047, 1.15291623,
                    1.25404929, 1.38181352, 1.51277828, 1.65615555, 1.81312174,
                    1.99784503, 2.18719569, 2.39449251, 2.6214364, 2.86988946,
                    3.16227766, 3.48445477, 3.79010887, 4.17625027, 4.54258822,
                    5.00539323, 5.51534944, 6.03808019, 6.65324794, 7.23686651,
                    7.97416822, 8.67365706, 9.55734093, 10.46316132, 11.5291623,
                    12.62186688, 13.81813522, 15.22594546, 16.66902172, 18.36728411,
                    20.10808843, 22.01388175, 24.10030129])

# J-factor appropriate for the HESS analysis
# As explained in 1103.3266 their ROI is defined by r<1 deg and |b|>0.3
# Also they use an Einasto profile with alpha=0.17, rs=20, rho0=0.39
# Then the average J factor over the ROI in TeV^2/cm^5

JHess = np.array([3.80113*10**20,5.21468*10**20,6.22573*10**20,6.97463*10**20,7.54186*10**20,7.97764*10**20,8.31518*10**20,8.57745*10**20])
