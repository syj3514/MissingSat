import numpy as np

IRandNumSize = 4
IBinarySize = 48
Mod4096DigitSize = 12
IZero = 0 
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
    
    k1 = j1
    k2 = j2 + k1 / 4096
    k3 = j3 + k2 / 4096
    k4 = j4 + k3 / 4096
    
    C = np.array([k1,k2,k3,k4])%4096
    return C

def ranf( Seed, RandNum ):
    RandNum = np.sum(Seed/Divisor)
    
    Outseed = ranfmodmult( Multiplier, Seed)
    return Outseed, RandNum

def nint(x):
    if x > 0: return np.floor(x + 0.5)
    else: return np.ceil(x - 0.5)


def poissdev(Seed, AverNum, PoissNum):
    RandNum = 0
    if(AverNum <= NPoissonLimit):
        Norm=np.exp(-AverNum) 
        Repar=1.0e0
        PoissNum=0
        Proba=1.0e0
        _, RandNum = ranf(Seed,RandNum)
        while(Repar*Norm <= RandNum and PoissNum <= 10*NPoissonLimit ):
            PoissNum=PoissNum+1
            Proba=Proba*AverNum/PoissNum
            Repar=Repar+Proba
    else:
        _, GaussNum = gaussdev(Seed,GaussNum)
        GaussNum=GaussNum*np.sqrt(AverNum)-0.5+AverNum
        if(GaussNum<=0.0e0): GaussNum=0.0e0
        PoissNum=nint(GaussNum)

    return PoissNum

v1, v2 = 0, 0
def gaussdev( Seed, GaussNum ):
    if (IGauss==IZero):
        rsq=0.0e0
        while (rsq>=1.0e0 or rsq<=0.0e0):
            Seed, v1 = ranf(Seed,v1)
            Seed, v2 = ranf(Seed,v2)
            v1=2.0e0*v1-1.0e0
            v2=2.0e0*v2-1.0e0
            rsq=v1**2+v2**2
        fac=np.sqrt(-2.0e0*np.log(rsq)/rsq)
        GaussBak=v1*fac
        GaussNum=v2*fac
        IGauss=1
    else:
        GaussNum=GaussBak
        IGauss=0
    return Seed, GaussNum