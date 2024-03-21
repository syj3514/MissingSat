_database = f"/home/jeon/MissingSat/database"
import matplotlib.pyplot as plt
from icl_tool import *
from icl_IO import mode2repo, pklsave, pklload
from icl_draw import fancy_axes
import os, glob
from IPython import get_ipython


from functools import lru_cache
def out2gyr(outs, snaps):
    table = snaps.iout_avail
    gyrs = np.zeros(len(outs))

    iout_table = table['iout']
    gyr_table = table['age']
    @lru_cache(None)
    def gyrfromout(iout):
        arg = iout_table==iout
        return gyr_table[arg][0]
    
    for i, iout in enumerate(outs):
        gyrs[i] = gyrfromout(iout)#table[table['iout']==iout][0]['age']
    return gyrs

def extract_from_LG(LG):
    allsats = None; allsubs = None; states = None
    keys = list(LG.keys())
    for key in keys:
        sats = LG[key]['sats']; subs = LG[key]['subs']; real = LG[key]['real']
        dink = real[real['state']=='dink']['hid']
        ind = np.isin(subs['id'], dink)
        subs['dink'][ind] = True; subs['dink'][~ind] = False
        state = np.zeros(len(subs), dtype='<U7')
        state[ind] = 'dink'; state[~ind] = 'pair'
        
        upair = real[real['state']=='upair']['hid']
        ind = np.isin(subs['id'], upair)
        state[ind] = 'upair'

        allsats = sats if allsats is None else np.hstack((allsats, sats))
        allsubs = subs if allsubs is None else np.hstack((allsubs, subs))
        states = state if states is None else np.hstack((states, state))
    argsort = np.argsort(allsubs['id'])
    allsubs = allsubs[argsort]; states = states[argsort]
    dinks = allsubs[states == 'dink']
    pairs = allsubs[states == 'pair']
    upairs = allsubs[states == 'upair']
    for dink in dinks: assert dink['dink']
    for pair in pairs: assert ~pair['dink']
    for upair in upairs: assert ~upair['dink']
    return allsats, allsubs, states, dinks, pairs, upairs

def _ibox(h, factor=1, rname='r'):
    return np.array([[h['x']-factor*h[rname], h['x']+factor*h[rname]],
                        [h['y']-factor*h[rname], h['y']+factor*h[rname]],
                        [h['z']-factor*h[rname], h['z']+factor*h[rname]]])
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

def makemovie(fnames, pathout, fps=15):
    import cv2
    from tqdm import tqdm
    frame_array = []
    for path in tqdm( fnames, desc="Read images..." ): 
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
    print(f"Size: {size} with {fps} frame/sec")
    out = cv2.VideoWriter(pathout,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in tqdm(range(len(frame_array)), desc="Write output..."):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def showall(key, subid, LG=None):
    if(LG is None):
        LG = pklload(f"{_database}/LG")

    fig, axes = fancy_axes(aspect='equal', ncols=2, dpi=400)
    fig.set_facecolor("k")

    draw_zoomin(
        key, subid, axes,
        LGkwargs={'box':LG[key]['box'], 'mode':'all'},
        subkwargs={'clean':False, 'mode':'SDG'}
        )

    return fig, axes

def printdt(table):
    for iname in table.dtype.names:
        print(f"{iname:10s} {table[iname]}")

def draw_zoomin(key, subid, axes, LGkwargs={}, subkwargs={}):
    load_LGimg(key, ax=axes[0], **LGkwargs)
    _, box = load_subimg(subid, ax=axes[1], **subkwargs)
    rec = plt.Rectangle((box[0,0], box[1,0]), np.diff(box[0]), np.diff(box[1]),color='magenta', fill=False)
    axes[0].add_patch(rec)
    for pos in ['top', 'bottom', 'right', 'left']:
        axes[1].spines[pos].set_edgecolor('magenta')

def load_LGimg(key, ax=None, box=None, mode='S'):
    modes = {'S':'star', 'D':'dm', 'G':'cell', 'SDG':'all', 'all':'all'}
    suffix = modes[mode]
    path = f"{_database}/photo/00_LG_image/NH_{key:04d}_{suffix}.png"
    img = plt.imread(path)
    if(box is None):
        LG = pklload(f"{_database}/LG")
        box = LG[key]['box']
    if(ax is not None): ax.imshow(img, extent=box[:2].flatten())
    return img, box

def load_subimg(subid, ax=None, clean=False, mode='SDG'):
    name1 = 'clean' if(clean) else 'info'
    name2 = 'All' if(mode=='SDG')or(mode=='all') else mode
    path = f"{_database}/photo/gallery/{name1}/{name2}/NH_sub{subid:07d}.png"
    img = plt.imread(path)
    box = read_subbox(subid)
    if(ax is not None): ax.imshow(img, extent=box[:2].flatten())
    return img, box

def write_subbox(subid, key=None, newpos=None):
    dtype = [
        ("subid", int), 
        ("x1", float),("x2", float),
        ("y1", float),("y2", float),
        ("z1", float),("z2", float),
        ]
    path = f"{_database}/parts/insub/boxes.pickle"
    boxes = None
    if(os.path.exists(path)):
        boxes = pklload(path)
        if(isinstance(boxes, np.void)):
            if(boxes['subid']==subid): return boxes
        else:
            if(subid in boxes['subid']): return boxes
    newbox = np.zeros(1, dtype=dtype)[0]
    newbox['subid'] = subid
    if(newpos is None):
        
        if(key is None):
            flist = os.listdir(f"{_database}/parts/insub")
            flist = [f for f in flist if(f.endswith(f'{subid:07d}.pickle'))][0]
            key = int(flist.split('_')[2])

        dm = pklload(f"{_database}/parts/insub/nh_dm_{key:04d}_{subid:07d}.pickle")
        x1 = dm['x'].min(); x2 = dm['x'].max()
        y1 = dm['y'].min(); y2 = dm['y'].max()
        z1 = dm['z'].min(); z2 = dm['z'].max()
        star = pklload(f"{_database}/parts/insub/nh_star_{key:04d}_{subid:07d}.pickle")
        if(len(star)>0):
            x1 = min(x1, star['x'].min()); x2 = max(x2, star['x'].max())
            y1 = min(y1, star['y'].min()); y2 = max(y2, star['y'].max())
            z1 = min(z1, star['z'].min()); z2 = max(z2, star['z'].max())
        cell = pklload(f"{_database}/parts/insub/nh_cell_{key:04d}_{subid:07d}.pickle")
        if(len(cell)>0):
            x1 = min(x1, cell['x'].min()); x2 = max(x2, cell['x'].max())
            y1 = min(y1, cell['y'].min()); y2 = max(y2, cell['y'].max())
            z1 = min(z1, cell['z'].min()); z2 = max(z2, cell['z'].max())
    else:
        x1,x2,y1,y2,z1,z2 = newpos
    newbox['x1'] = x1; newbox['x2'] = x2
    newbox['y1'] = y1; newbox['y2'] = y2
    newbox['z1'] = z1; newbox['z2'] = z2

    boxes = newbox if(boxes is None) else np.hstack((boxes, newbox))
    pklsave(boxes, path, overwrite=True)
    return boxes

def read_subbox(subid):
    path = f"{_database}/parts/insub/boxes.pickle"
    boxes = write_subbox(subid)

    where = np.where(boxes['subid']==subid)[0][0]
    x1 = boxes['x1'][where]; x2 = boxes['x2'][where]
    y1 = boxes['y1'][where]; y2 = boxes['y2'][where]
    z1 = boxes['z1'][where]; z2 = boxes['z2'][where]
    return np.array([[x1,x2],[y1,y2],[z1,z2]])


def calc_rhalf(gal, part, weights, ratio=0.5):
    dist = distance(gal, part)
    argsort = np.argsort(dist)
    sw = np.cumsum(weights[argsort])
    sw /= sw[-1]
    return dist[argsort][np.argmin(np.abs(sw-ratio))]

def calc_rhalf_sorted(sorted_dist, sorted_weights, ratio=0.5):
    sw = np.cumsum(sorted_weights)
    sw /= sw[-1]
    return sorted_dist[np.argmin(np.abs(sw-ratio))]

def calc_tform(part, weights, ratio=0.5):
    age = part['age','Gyr']
    argsort = np.argsort(age)
    sw = np.cumsum(weights[argsort])
    sw /= sw[-1]
    return age[argsort][np.argmin(np.abs(sw-ratio))]






# Test Branch
# def massbranch(branch):
#     val = np.log10(branch['mvir'])
#     firstm = val[0]
#     mask1 = np.full(len(branch), True, dtype=bool)
#     for i in range(len(mask1)):
#         if(val[i] < 9): continue
#         arr = val[max(0,i-100):i+100]
#         mask1[i] = val[i] <= min( (np.mean(arr) + 4*np.std(arr)), firstm+3 )
#     upper = (np.median(val) + 3*np.std(val))
#     if(upper>8.5):
#         mask2 = val <= upper
#         return mask1&mask2
#     return mask1
def _mass_local(logmvir, oldmask):
    mask_local = np.full(len(logmvir), True, dtype=bool)*oldmask
    for i in range(len(mask_local)):
        if(~oldmask[i]): continue
        if(logmvir[i] < 9): continue
        arr = logmvir[max(0,i-100):i+100]
        mask_local[i] = logmvir[i] <= min( (np.mean(arr) + 4*np.std(arr)), logmvir[0]+3 )
    return mask_local

def massbranch(branch):
    oldmask = np.full(len(branch), True)
    val = np.log10(branch['mvir'])
    firstm = val[0]
    if(np.max(val) < firstm+1.5):
        return oldmask
    else:
        mask_local = _mass_local(val, oldmask)
        a = np.median(val[oldmask]); b = np.std(val[oldmask]); err = min(3*b, 3)
        mask_global = (val <= (a + err))&(oldmask)
        newmask = mask_local&mask_global
        while(np.sum(oldmask) != np.sum(newmask)):
            oldmask = oldmask&newmask
            mask_local = _mass_local(val, oldmask)
            a = np.median(val[oldmask]); b = np.std(val[oldmask]); err = min(3*b, 3)
            if(np.max(val[oldmask]) < a+2): break
            mask_global = (val <= (a + err))&(oldmask)
            newmask = mask_local&mask_global
        mmask = oldmask&newmask
    return mmask

def velbranch(branch, snaps):
    iout = snaps.iout_avail['iout']
    fsnap = snaps.get_snap(iout[-1]); unitl_com = fsnap.unit_l/fsnap.aexp
    aexp = snaps.iout_avail['aexp']
    age = snaps.iout_avail['age']
    mask = np.full(len(branch), False, dtype=bool)
    mask[0] = True
    factor = 1
    for i in range(len(mask)-1):
        if(mask[i]):
            nb = branch[i]
            niout = nb['timestep']; nwhere = np.where(iout == niout)[0][0]; nage = age[nwhere]
            unit_l = unitl_com * aexp[nwhere]
        pb = branch[i+1]
        piout = pb['timestep']; pwhere = np.where(iout == piout)[0][0]; page = age[pwhere]
        dt = (nage - page)*1e9 * 365*24*3600 # sec
        dx = (nb['vx']*dt) * 1e5 # cm
        dy = (nb['vy']*dt) * 1e5
        dz = (nb['vz']*dt) * 1e5
        nnx = nb['x'] - dx/unit_l
        nny = nb['y'] - dy/unit_l
        nnz = nb['z'] - dz/unit_l
        dist2 = np.sqrt( (nnx-pb['x'])**2 + (nny-pb['y'])**2 + (nnz-pb['z'])**2 )
        if(dist2 < factor*(nb['rvir']+pb['rvir'])) and (pb['mvir'] < nb['mvir']*1e2):
            mask[i+1] = True
            factor = 1
        else:
            factor += 0.01
    return mask

def polybranch(branch, vmask=None, return_poly=False):
    if(vmask is None): vmask = np.full(len(branch), 0)
    score = (branch['take_score']*branch['give_score']) * (vmask+0.5)
    polyx = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['x'], 20, w=score)
    resix = branch['x'] - polyx(branch['timestep'])
    stdx = np.std(resix)
    polyy = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['y'], 20, w=score)
    resiy = branch['y'] - polyy(branch['timestep'])
    stdy = np.std(resiy)
    polyz = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['z'], 20, w=score)
    resiz = branch['z'] - polyz(branch['timestep'])
    stdz = np.std(resiz)

    resi = np.sqrt(resix**2 + resiy**2 + resiz**2)
    where1 = (resi > np.sqrt(stdx**2 + stdy**2 + stdz**2))
    where2 = resi/np.sqrt(3) > 1e-4
    where = where1&where2
    if(return_poly):
        return (~where), polyx, polyy, polyz
    return (~where)