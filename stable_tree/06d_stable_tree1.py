from IPython import get_ipython


def type_of_script():
    """
    Detects and returns the type of python kernel
    :return: string 'jupyter' or 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


if type_of_script() == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

import numpy as np
import os, glob, atexit, signal
os.nice(19)
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from rur.sci.geometry import get_angles, euler_angle
from rur.utool import rotate_data
from scipy.ndimage import gaussian_filter
uri.timer.verbose=0
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle
from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value








mode1 = 'nh'
database1 = f"/home/jeon/MissingSat/database/{mode1}"
iout1 = 1026
repo1, rurmode1, dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, iout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']; nout=nout1[nout1 <= iout1]
gal1s = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hal1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)


from common_func import *


LG1 = pklload(f"{database1}/LocalGroup.pickle")
allsubs1 = None
states1 = None
for key in LG1.keys():
    subs = LG1[key]['subs']
    real = LG1[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True
    subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'
    state[~ind] = 'pair'
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsubs1 = subs if allsubs1 is None else np.hstack((allsubs1, subs))
    states1 = state if states1 is None else np.hstack((states1, state))
argsort = np.argsort(allsubs1['id'])
allsubs1 = allsubs1[argsort]
states1 = states1[argsort]


##########################################################################
print("NH")
tree1 = pklload(f"{database1}/stable_progenitors.pickle")
fnames1 = os.listdir(f"{database1}/stable_prog")
fnames1 = [fname for fname in fnames1 if(fname.startswith('subhalos'))]
fnames1.sort(reverse=True)

tmp = pklload(f"{database1}/stable_prog/{fnames1[0]}")[0] # 1026 snapshot
where = np.where(isin(tmp['lastid'], allsubs1['id']))[0]
tmp = tmp[where]
trees1 = np.empty(tmp.shape[0]*len(fnames1), dtype=tmp.dtype)
cursor = 0
uri.timer.verbose=0
for fname in tqdm( fnames1 ):
    tmp = pklload(f"{database1}/stable_prog/{fname}")[0]
    if(len(tmp) > len(allsubs1)): tmp = tmp[where]
    trees1[cursor : cursor+len(tmp)] = tmp
    cursor += len(tmp)
trees1 = trees1[trees1['timestep'] <= iout1]

stable_tree1 = {}
for i, sub in tqdm(enumerate(allsubs1), total=len(allsubs1)):
    arg = np.where(allsubs1['id']==sub['id'])[0][0]
    branch = trees1[arg::len(allsubs1)]#[trees['lastid'] == target['id']]
    branch = branch[branch['take_score']>0]

    score = branch['give_score']*branch['take_score']/branch['aexp']*snap1['aexp']
    umask = (score > 0.1) & (branch['take_score']>0.05)# | ((branch['give_score']+branch['take_score']) > 1.1) | (branch['take_score'] > 0.4)

    dx = np.abs(np.diff(branch['x']))
    meandx = np.mean(dx)
    dy = np.abs(np.diff(branch['y']))
    meandy = np.mean(dy)
    dz = np.abs(np.diff(branch['z']))
    meandz = np.mean(dz)
    dp = np.sqrt(dx**2 + dy**2 + dz**2)
    meandp = np.sqrt(meandx**2 + meandy**2 + meandz**2)*3
    pmask = dp <= meandp
    pmask = np.insert(pmask, 0, True)

    first = np.min(branch['timestep'][umask & pmask])

    cbranch = branch[branch['timestep'] >= first]
    top20per = top20per = np.argsort(branch['take_score'])[-int(len(branch)/5):]
    if(len(top20per)>10):
        m16,m50,m84 = np.percentile(branch['mdm_vir'][top20per], q=[16,50,84])
        lower = m16/20; upper = m84*20
        if(lower > branch[0]['mdm_vir']): lower = branch[0]['mdm_vir']
        if(upper < branch[0]['mdm_vir']): upper = branch[0]['mdm_vir']
    else:
        lower = branch[0]['mdm_vir']/20; upper = branch[0]['mdm_vir']*20

    mbranch = cbranch[(cbranch['mdm_vir'] <= upper)&(cbranch['mdm_vir'] >= lower)&(cbranch['take_score']>0.01)]
    stable_tree1[sub['id']] = mbranch
pklsave(stable_tree1,f"{database1}/stable_tree.pickle", overwrite=True)