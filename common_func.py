database = f"/home/jeon/MissingSat/database"
import matplotlib.pyplot as plt
from icl_tool import *
from icl_IO import mode2repo, pklsave, pklload
from icl_draw import fancy_axes
import os, glob

def showall(key, subid, LG=None):
    if(LG is None):
        LG = pklload(f"{database}/LG")

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
    path = f"{database}/photo/00_LG_image/NH_{key:04d}_{suffix}.png"
    img = plt.imread(path)
    if(box is None):
        LG = pklload(f"{database}/LG")
        box = LG[key]['box']
    if(ax is not None): ax.imshow(img, extent=box[:2].flatten())
    return img, box

def load_subimg(subid, ax=None, clean=False, mode='SDG'):
    name1 = 'clean' if(clean) else 'info'
    name2 = 'All' if(mode=='SDG')or(mode=='all') else mode
    path = f"{database}/photo/gallery/{name1}/{name2}/NH_sub{subid:07d}.png"
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
    path = f"{database}/parts/insub/boxes.pickle"
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
            flist = os.listdir(f"{database}/parts/insub")
            flist = [f for f in flist if(f.endswith(f'{subid:07d}.pickle'))][0]
            key = int(flist.split('_')[2])

        dm = pklload(f"{database}/parts/insub/nh_dm_{key:04d}_{subid:07d}.pickle")
        x1 = dm['x'].min(); x2 = dm['x'].max()
        y1 = dm['y'].min(); y2 = dm['y'].max()
        z1 = dm['z'].min(); z2 = dm['z'].max()
        star = pklload(f"{database}/parts/insub/nh_star_{key:04d}_{subid:07d}.pickle")
        if(len(star)>0):
            x1 = min(x1, star['x'].min()); x2 = max(x2, star['x'].max())
            y1 = min(y1, star['y'].min()); y2 = max(y2, star['y'].max())
            z1 = min(z1, star['z'].min()); z2 = max(z2, star['z'].max())
        cell = pklload(f"{database}/parts/insub/nh_cell_{key:04d}_{subid:07d}.pickle")
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
    path = f"{database}/parts/insub/boxes.pickle"
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