import numpy as np

#############################################################
#
#   Common Constant
#
#############################################################
# From Namelist
n_gmc = 10 # Density above which Hopkins and Cen SF routines are evaluated:
omega_b=0.0455 # Omega Baryon
gamma=1.6666667
n_star=10 # Star formation density threshold in H/cc

# `amr_parameters.f90`
del_star=2.e2 # Minimum overdensity to define ISM
nsn2mass = -1 # Star particle mass in units of the number of SN:
fstar_min = 1 # Mstar,min = nH*dx_min^3*fstar_min
m_star = -1 # Star particle mass in units of mass_sph

# `hydro_parameters.f90`
smallr = 1e-10
smallc = 1e-10

# `cooling_module.f90`
rhoc = 1.8800000e-29
XH = 0.76
mH = 1.6600000e-24

# `star_formation.f90`
''' proto stellar feedback parameter 
that controls the actual amount of gas 
above scrit that is able to form stars'''
e_cts = 0.5  # would be 1.0 without feedback (Federrath & Klessen 2012)
eps_star=0.5             #base SF efficiency (was not settable in NH!)
'''empirical parameters of the model 
determined by the best-fit values 
between the theory and the numerical experiments
(Federrath & Klessen 2012).'''
phi_t = 0.57; theta = 0.33 # best fit values of the Padoan & Nordlund multi-scale sf model to GMC simulation data 




#############################################################
#
#   Common Variable
#
#############################################################
# From raw
#   `pario/output_amr_io.f90`
param_where = {
    'dt_old' : ('amr', 12, 'r'),
    'dt_new' : ('amr', 13, 'r'),
    'mass_sph' : ('amr', 18, 'r'),
    'localseed' : ('part', 3, 'i')
}

from rur.fortranfile import FortranFile
def params(key, isnap, icpu=1):
    if(key in param_where):
        return from_raw(key, isnap, icpu=icpu)
    else:
        return from_snap(key, isnap)

def from_raw(key, isnap, icpu=1):
    kind, skip, dtype = param_where[key]
    with FortranFile(f"{isnap.path}/{kind}_{isnap.iout:05d}.out{icpu:05d}") as f:
        f.skip_records(skip)
        v = f.read_reals() if(dtype=='r') else f.read_ints()
    return v

def from_snap(key, isnap):
    try:
        # h0 = isnap.H0 # Hubble constant in km/s/Mpc
        # aexp = isnap.aexp # Current expansion factor
        # omega_m = isnap.omega_m # Omega Matter
        if(key=='h0'): return isnap.params['H0']
        return isnap.params[key]
    except:
        if(key=='scale_nH'):
            return XH/mH * isnap.unit_d # `units.f90`
        elif(key=='nCOM'):
            h0 = from_snap('h0', isnap)
            aexp = from_snap('aexp', isnap)
            return del_star*omega_b*rhoc*(h0/100.)**2/aexp**3*XH/mH # `star_formation.f90`
        elif(key=='d_gmc'):
            nCOM = from_snap('nCOM', isnap)
            scale_nH = from_snap('scale_nH', isnap)
            return max(nCOM, n_gmc) / scale_nH # `star_formation.f90`
        elif(key=='factG'):
            omega_m = from_snap('omega_m', isnap)
            aexp = from_snap('aexp', isnap)
            return 3/8/np.pi*omega_m*aexp # `hydro_flag.f90`

#############################################################
#
#   random.f90
#
#############################################################
IRandNumSize = 4
IBinarySize = 48
Mod4096DigitSize = 12
NPoissonLimit = 10

Multiplier = np.array([373, 3707, 1442, 647])
DefaultSeed = np.array([3281, 4041, 595, 2376])
Divisor = np.array([281474976710656.0,68719476736.0,16777216.0,4096.0])
IGauss = 0
GaussBak = 0.0

def ranfmodmult( A, B ):
    j1 = A[0]*B[0]
    j2 = A[0]*B[1] + A[1]*B[0]
    j3 = A[0]*B[2] + A[1]*B[1] + A[2]*B[0]
    j4 = A[0]*B[3] + A[1]*B[2] + A[2]*B[1] + A[3]*B[0]
    
    k1 = int(j1)
    k2 = int(j2 + k1 / 4096)
    k3 = int(j3 + k2 / 4096)
    k4 = int(j4 + k3 / 4096)
    
    C = np.array([k1,k2,k3,k4])%4096
    return C

def ranf( Seed, RandNum ):
    RandNum = np.sum(Seed/Divisor)
    
    Outseed = ranfmodmult( Multiplier, Seed)
    Seed = Outseed
    return Outseed, RandNum

def nint(x):
    if x > 0: return np.floor(x + 0.5)
    else: return np.ceil(x - 0.5)


def poissdev(Seed, AverNum, PoissNum):
    RandNum = 0
    if(AverNum <= NPoissonLimit):
        Norm=np.exp(-AverNum) 
        Repar=1.0
        PoissNum=0
        Proba=1.0
        Seed, RandNum = ranf(Seed,RandNum)
        while(Repar*Norm <= RandNum and PoissNum <= 10*NPoissonLimit ):
            PoissNum += 1
            Proba *= AverNum/PoissNum
            Repar += Proba
    else:
        _, GaussNum = gaussdev(Seed,GaussNum)
        GaussNum=GaussNum*np.sqrt(AverNum)-0.5+AverNum
        if(GaussNum<=0.0): GaussNum=0.0
        PoissNum=nint(GaussNum)

    return PoissNum


def gaussdev( Seed, GaussNum ):
    global IGauss, GaussBak
    v1, v2 = 0, 0
    if (IGauss==0):
        rsq=0.0
        while (rsq>=1.0 or rsq<=0.0):
            Seed, v1 = ranf(Seed,v1)
            Seed, v2 = ranf(Seed,v2)
            v1 = 2*v1 - 1
            v2 = 2*v2 - 1
            rsq = v1**2 + v2**2
        fac = np.sqrt(-2.0*np.log(rsq)/rsq)
        GaussBak = v1*fac
        GaussNum = v2*fac
        IGauss=1
    else:
        GaussNum = GaussBak
        IGauss=0
    return Seed, GaussNum

#############################################################
#
#   star_formation.f90
#
#############################################################
def cell2uold(cell, gamma=1.6666667, smallr=1e-10):
    uold1 = cell['rho']
    d = np.where(uold1>smallr, uold1, smallr)
    uold2 = cell['vx']*d
    uold3 = cell['vy']*d
    uold4 = cell['vz']*d
    uold5 = cell['P']/(gamma-1) + uold4**2/2/d + uold3**2/2/d + uold2**2/2/d
    uold = np.vstack((uold1, uold2, uold3, uold4, uold5)).T
    return uold

def uc(cell, key, gamma=1.6666667, smallr=1e-10):
    '''
    Convert hydro data in `Primitive format` to `Conservative format`
    - uold1: density
    - uold2: vx * density = momentum_x
    - uold3: vy * density = momentum_y
    - uold4: vz * density = momentum_z
    - uold5: total energy: (P / (gamma-1)) + (vz**2 / 2 / d) + (vy**2 / 2 / d) + (vx**2 / 2 / d)
    '''
    uold1 = cell['rho']
    if(key=='rho'):
        return uold1
    d = np.where(uold1>smallr, uold1, smallr)
    if(key[0]=='v'):
        return cell[key]*d
    uold2 = cell['vx']*d
    uold3 = cell['vy']*d
    uold4 = cell['vz']*d
    if(key=='P'):
        return cell[key]/(gamma-1) + 0.5*(uold4**2 + uold3**2 + uold2**2)/d
    else:
        return None

def erfc(x):
    pv= 1.26974899965115684e+01; ph= 6.10399733098688199e+00
    p0= 2.96316885199227378e-01; p1= 1.81581125134637070e-01
    p2= 6.81866451424939493e-02; p3= 1.56907543161966709e-02
    p4= 2.21290116681517573e-03; p5= 1.91395813098742864e-04
    p6= 9.71013284010551623e-06; p7= 1.66642447174307753e-07
    q0= 6.12158644495538758e-02; q1= 5.50942780056002085e-01
    q2= 1.53039662058770397e+00; q3= 2.99957952311300634e+00
    q4= 4.95867777128246701e+00; q5= 7.41471251099335407e+00
    q6= 1.04765104356545238e+01; q7= 1.48455557345597957e+01

    y = x**2
    y = np.exp(-y) * x * ( p7/(y+q7)+p6/(y+q6) + p5/(y+q5)+p4/(y+q4)+p3/(y+q3) + p2/(y+q2)+p1/(y+q1) + p0/(y+q0))
    if (x < ph): y = y+2/(np.exp(pv*x)+1)
    return y
def erfcs(xs):
    pv= 1.26974899965115684e+01; ph= 6.10399733098688199e+00
    p0= 2.96316885199227378e-01; p1= 1.81581125134637070e-01
    p2= 6.81866451424939493e-02; p3= 1.56907543161966709e-02
    p4= 2.21290116681517573e-03; p5= 1.91395813098742864e-04
    p6= 9.71013284010551623e-06; p7= 1.66642447174307753e-07
    q0= 6.12158644495538758e-02; q1= 5.50942780056002085e-01
    q2= 1.53039662058770397e+00; q3= 2.99957952311300634e+00
    q4= 4.95867777128246701e+00; q5= 7.41471251099335407e+00
    q6= 1.04765104356545238e+01; q7= 1.48455557345597957e+01

    ys = xs**2
    ys = np.exp(-ys) * xs * ( p7/(ys+q7)+p6/(ys+q6) + p5/(ys+q5)+p4/(ys+q4)+p3/(ys+q3) + p2/(ys+q2)+p1/(ys+q1) + p0/(ys+q0) )
    where = (xs < ph)
    ys[where] = ys[where]+2/(np.exp(pv*xs[where])+1)
    return ys
