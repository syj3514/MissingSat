import numpy as np
from icl_tool import *
from tqdm import tqdm

def make_banlist(gals, hals):
    """
    Generate a ban list based on distance comparisons between galaxies and halos.

    Parameters
    ----------
    gals : numpy.ndarray
        Array containing galaxy information.
    hals : numpy.ndarray
        Array containing halo information.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating banned halos.
    """
    ban = np.zeros(len(hals), dtype=bool)
    for i,hal in tqdm(enumerate(hals), total=len(hals)):
        dists = distance(gals, hal)
        if(np.any(dists  <  gals['r']+hal['r'])):
            pass
        else:
            ban[i] = True
    return ban

def cutting(hal, gals, hban=[],gban=[], lvlmin=0, lvlmax=1):
    """
    Apply various cuts to select members around a halo.

    This function performs a series of cuts on galaxy members around a given halo to select potential
    member galaxies. The cuts are based on different criteria, such as halo ID bans, halo level,
    distance from halo center, and more.

    Parameters
    ----------
    hal : numpy.ndarray
        Array containing information about the halo.
    gals : numpy.ndarray
        Array containing information about galaxies.
    hban : list, optional
        List of banned halo IDs, by default [].
    gban : list, optional
        List of banned galaxy IDs, by default [].
    lvlmin : int, optional
        Minimum halo level for selection, by default 0.
    lvlmax : int, optional
        Maximum halo level for selection, by default 1.

    Returns
    -------
    int, int
        ID of the selected galaxy member, and a code indicating the applied cut:
        - 0: No suitable members after cuts.
        - 1: Only one suitable member based on initial cuts.
        - 2: Cut based on halo's virial radius.
        - 3: Cut based on galaxy member's halo level.
        - 4: Cut based on distance from halo center.
        - 5: Cut based on galaxy member's mass.

    """
    if(hal['id'] in hban): return 0, 0
    if( (hal['level']<lvlmin) or (hal['level']>lvlmax) ): return 0, 0

    # Initial cut
    if(len(gban)>0):
        mems = gals[np.isin(gals['id'], gban, invert=True, assume_unique=True)]
    else:
        mems = gals
    if(len(mems)==0): return 0,0

    dists = distance(mems, hal)
    mems = mems[dists < (mems['r']+hal['rvir'])]
    if(len(mems)==0): return 0,0
    if(len(mems)==1): return mems[0]['id'],1

    # Cut by rvir
    temp = cut_sphere(mems, hal['x'], hal['y'], hal['z'], hal['rvir'])
    mems = temp if(len(temp)>0) else mems
    if(len(mems)==1): return mems[0]['id'],2

    # Cut by lvl
    lvlcut = 0
    temp = mems[mems['level']<=lvlcut]
    while(len(temp)==0):
        lvlcut += 1
        temp = mems[mems['level']<=lvlcut]
    if(len(mems)==1): return mems[0]['id'],3

    # Cut by halo center
    dists = distance(mems, hal)
    temp = mems[dists < mems['r']]
    mems = temp if(len(temp)>0) else mems
    if(len(mems)==1): return mems[0]['id'],4

    # Cut by mass
    return mems[np.argmax(mems['m'])]['id'],5


def find_cengal_of_lvl1hals(gals, hals, bans):
    """
    Locate central galaxies within level 1 halos while accounting for constraints and competition.

    Parameters:
    -----------
    gals : ndarray
        Galaxy dataset.
    hals : ndarray
        Halo dataset.
    bans : ndarray
        Array of banned halo indices.

    Returns:
    --------
    ndarray, ndarray
        Arrays of central galaxy IDs and corresponding scores for level 1 halos.
    """
    gal_of_hals = np.zeros(len(hals), dtype=int)
    scores = np.zeros(len(hals), dtype=int)
    hban = np.where(bans)[0]+1
    for i, hal in tqdm(enumerate(hals), total=len(hals)):
        if(bans[i]): continue
        if(hal['level'] != 1): continue
        gid, score = cutting(hal, gals, hban=hban, lvlmin=1, lvlmax=1)
        if(gid>0):
            if(gid in gal_of_hals):
                where = np.where(gal_of_hals==gid)[0][0]
                # Competition
                if(score > scores[where]):
                    gal_of_hals[where] = -gid
                    scores[where] = -gid
                    gal_of_hals[i] = gid
                    scores[i] = score
                elif(score < scores[where]):
                    pass
                else:
                    check_order(gals['id'])
                    gal = gals[gid-1]
                    my_dist = distance(gal, hal)
                    other_dist = distance(gal, hals[where])
                    if(my_dist < other_dist):
                        gal_of_hals[where] = -gid
                        scores[where] = -gid
                        gal_of_hals[i] = gid
                        scores[i] = score
                    else:
                        pass
            else:
                gal_of_hals[i] = gid
                scores[i] = score
    return gal_of_hals, scores


def find_cengal_of_others(hals, gals, bans, result):
    """
    Find central galaxies of halos considering competition and selection criteria.

    This function identifies central galaxies for halos by iterating over a list of halos and
    applying selection cuts on potential galaxy members based on various criteria. The selection
    takes into account previously selected galaxies and competition among halos for common galaxies.

    Parameters
    ----------
    hals : numpy.ndarray
        Array containing information about halos.
    gals : numpy.ndarray
        Array containing information about galaxies.
    bans : numpy.ndarray
        Boolean array indicating banned halos.
    result : numpy.ndarray
        Array containing previously selected galaxy-halo associations.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        An array containing IDs of central galaxies selected for each halo, and an array of scores
        corresponding to each central galaxy selection.

    Notes
    -----
    - This function internally uses the `cutting` function to perform cuts for galaxy selection.
    - The selection process considers competition among halos and prioritizes central galaxy
      selection based on various criteria.

    """
    gal_of_hals = np.zeros(len(hals), dtype=int)
    scores = np.zeros(len(hals), dtype=int)

    already = result[result['halo_id'] > 0]
    hban = already['halo_id']
    gban = already['id']
    for i, hal in tqdm(enumerate(hals), total=len(hals)):
        if(bans[i]): continue
        if(hal['id'] in already['halo_id']): continue
        gid, score = cutting(hal, gals, hban=hban, gban=gban, lvlmin=0, lvlmax=10)
        if(gid>0):
            if(gid in gal_of_hals):
                where = np.where(gal_of_hals==gid)[0][0]
                # Competition
                if(score > scores[where]):
                    gal_of_hals[where] = -gid
                    scores[where] = -gid
                    gal_of_hals[i] = gid
                    scores[i] = score
                elif(score < scores[where]):
                    pass
                else:
                    check_order(gals['id'])
                    gal = gals[gid-1]
                    my_dist = distance(gal, hal)
                    other_dist = distance(gal, hals[where])
                    if(my_dist < other_dist):
                        gal_of_hals[where] = -gid
                        scores[where] = -gid
                        gal_of_hals[i] = gid
                        scores[i] = score
                    else:
                        pass
            else:
                gal_of_hals[i] = gid
                scores[i] = score
    return gal_of_hals, scores


def find_halos_for_other(gal, hals, results):
    """
    Find suitable halos for a given galaxy considering various selection criteria.

    This function identifies potential halos for a given galaxy based on different criteria, including
    distance considerations, neighboring halos, and halo hierarchy levels. It returns the distance to
    the selected halo, the halo's ID, whether the halo is considered central, and whether it's a main halo.

    Parameters
    ----------
    gal : numpy.ndarray
        Array containing information about the galaxy.
    hals : numpy.ndarray
        Array containing information about halos.
    results : numpy.ndarray
        Array containing results of previous galaxy-halo associations.

    Returns
    -------
    float, int, bool, bool
        Distance to the selected halo, halo ID, whether the halo is considered central, and whether it's a main halo.

    Notes
    -----
    - This function internally uses the `cut_sphere` function for spatial cuts.
    - The selection process considers various factors, including neighboring halos, halo hierarchy levels,
      and previously associated results.

    """
    check_order(hals['id'])
    check_order(results['id'])
    # return halo_id, central, main
    dist, hid, central, main = 1, 0, False, False
    host = None
    hosts = None
    key = 'hostsub' if(gal['hostsub']>0) else 'host'
    if(gal['level']>1):
        host = results[gal[key]-1]
        while(host['halo_id']<=0):
            host = results[host[key]-1]
            if(host['level']==1):
                break
    else:
        dists = distance(results, gal)
        hosts = results[dists < results['r']]
        hosts = hosts[hosts['id'] != gal['id']]
        if(len(hosts)>0):
            dists = distance(hosts, gal)
            host = hosts[np.argmin(dists)]
            while(host['halo_id']<=0):
                
                host = results[host[key]-1]
                if(host['level']==1):
                    break
    
    thalos = cut_sphere(hals, gal['x'], gal['y'], gal['z'], gal['r'])
    hids = thalos['id']
    if(host is not None):
        hids = np.append(hids, host['halo_id'])
    if(hosts is not None):
        hids = np.append(hids, hosts['halo_id'])
    hids = hids[hids>0]
    hids = np.unique(hids)
    thalos = hals[hids-1]
    if(len(thalos)==0): return dist, hid, central, main

    gals_of_hals = [[]] * len(thalos)
    for i, thalo in enumerate(thalos):
        if(thalo['id'] in results['halo_id']):
            gals_of_hals[i] = np.where(results['halo_id']==thalo['id'])[0]
    lengs = np.array([len(iarr) for iarr in gals_of_hals])
    wheres = np.where(lengs==0)[0]
    if(len(wheres)==0):
        # All halos have galaxies
        if(host is not None):
            
            hid = host['halo_id']
            
            thalo = hals[hid-1]
            dist = distance(thalo, gal)
            central = False
            main = host['main']
        else:
            pass
    elif(len(wheres)==1):
        # Only one halo has no galaxy
        thalo = thalos[wheres[0]]
        dist = distance(thalo, gal)
        hid = thalo['id']
        central = True
        main = (thalo['level']==1)
    else:
        # Multiple halos have no galaxy
        if(host is not None):
            hid = host['halo_id']
            thalo = hals[hid-1]
            dist = distance(thalo, gal)
            central = False
            main = host['main']
        else:
            dists = distance(thalos, gal)
            thalo = thalos[np.argmin(dists)]
            dist = dists[np.argmin(dists)]
            hid = thalo['id']
            central = True
            main = (thalo['level']==1)
    return dist, hid, central, main


def find_halos_for_others(hals, results):
    """
    Find suitable halos for a list of galaxies considering various selection criteria.

    This function identifies potential halos for a list of galaxies based on different criteria,
    including neighboring halos, halo hierarchy levels, and previous associations. It returns an
    array containing information about the selected halos for each galaxy.

    Parameters
    ----------
    hals : numpy.ndarray
        Array containing information about halos.
    results : numpy.ndarray
        Array containing results of previous galaxy-halo associations.

    Returns
    -------
    numpy.ndarray
        Array containing selected halo information for each galaxy, including distance to the halo,
        halo ID, whether the halo is considered central, and whether it's a main halo.

    Notes
    -----
    - This function internally uses the `find_halos_for_other` function for individual galaxy-halo selection.
    - The selection process considers various factors, including neighboring halos, halo hierarchy levels,
      and previously associated results.

    """
    argsort = np.lexsort((results['id'], -results['m'], results['level']))
    gals = results[argsort]
    arr = np.empty(len(gals), dtype=[('id','i4'),('m','f8'),('level','i4'),('dist','f8'), ('halo_id', 'i4'), ('central', 'bool'), ('main', 'bool')])
    arr['id'] = -1
    for ith in tqdm(range(len(gals))):
        gal = gals[ith]
        if(gal['halo_id']>0): continue
        dist, hid, central, main = find_halos_for_other(gal, hals, results)#gal['id']==1024 )
        arr[ith]['id'] = gal['id']
        arr[ith]['m'] = gal['m']
        arr[ith]['level'] = gal['level']
        arr[ith]['dist'] = dist
        arr[ith]['halo_id'] = hid
        arr[ith]['central'] = central
        arr[ith]['main'] = main
    return arr[(arr['id']>0)&(arr['halo_id']>0)]

def find_halos_for_otherss(hals, results):
    """
    Assign selected halos to galaxies based on specific criteria and associations.

    This function assigns selected halos to galaxies considering central galaxy selection criteria and
    previously established associations. It updates the `halo_id`, `central`, and `main` fields of the
    galaxy array with appropriate halo information.

    Parameters
    ----------
    hals : numpy.ndarray
        Array containing information about halos.
    results : numpy.ndarray
        Array containing results of previous galaxy-halo associations.

    Returns
    -------
    numpy.ndarray
        Array containing updated galaxy information, including assigned halo IDs, centrality status,
        and main halo indication.

    Notes
    -----
    - This function internally uses the `find_halos_for_others` function for halo selection.
    - The selection process prioritizes central galaxy assignment based on specific criteria and
      updates the galaxy array accordingly.

    """
    arr = find_halos_for_others(hals, results)
    gals = np.copy(results)
    check_order(gals['id'])

    unique, count = np.unique(arr['halo_id'], return_counts=True)
    for uni, cou in tqdm(zip(unique, count), total=len(unique)):
        insts = arr[arr['halo_id'] == uni]
        if(uni in gals['halo_id']):
            # Already central in `gals`
            for inst in insts:
                gals[inst['id']-1]['halo_id'] = uni
                gals[inst['id']-1]['central'] = inst['central']
                gals[inst['id']-1]['main'] = inst['main']
        else:
            # Not central in `gals`
            insts = arr[arr['halo_id'] == uni]
            if(np.sum(insts['central'])>1):
                pass
            elif(np.sum(insts['central'])==0):
                # No centrals
                pass
            else:
                # One central
                inst = insts[0]
                gals[inst['id']-1]['halo_id'] = uni
                gals[inst['id']-1]['central'] = inst['central']
                gals[inst['id']-1]['main'] = inst['main']
    return gals


def final_job(hals, results):
    """
    Perform final assignment of halos to galaxies based on neighboring and selection criteria.

    This function completes the halo assignment process for galaxies by considering nearby galaxies
    and applying specific selection criteria. It updates the `halo_id`, `central`, `main`, `dist`,
    `fcontam`, and other fields of the galaxy array with appropriate information.

    Parameters
    ----------
    hals : numpy.ndarray
        Array containing information about halos.
    results : numpy.ndarray
        Array containing results of previous galaxy-halo associations.

    Returns
    -------
    numpy.ndarray
        Array containing updated galaxy information, including assigned halo IDs, centrality status,
        main halo indication, distance to halo, fractional contamination, and other halo-related information.

    Notes
    -----
    - This function internally uses the `cut_sphere` function for spatial cuts and selection criteria.
    - The assignment process considers neighboring galaxies and applies specific rules to determine
      the best-fitting halo for each galaxy.

    """
    gals = np.copy(results)
    for gal in gals:
        if(gal['halo_id']>0): continue
        neighbors = cut_sphere(results, gal['x'], gal['y'], gal['z'], gal['r'], both_sphere=True, rname='r')
        if(len(neighbors)>0):
            dists = distance(neighbors, gal)
            ineigh = neighbors[ np.argmin(dists) ]
            halo_id = ineigh['halo_id']
            if(halo_id>0):
                gal['halo_id'] = halo_id
                gal['central'] = False
                gal['main'] = ineigh['main']
                continue
        ihals = cut_sphere(gals, gal['x'], gal['y'], gal['z'], gal['r'], both_sphere=True, rname='r')
        if(len(ihals)>0):
            dists = distance(ihals, gal)
            ihal = ihals[ np.argmin(dists) ]
            gal['halo_id'] = ihal['id']
            gal['central'] = False if(ihal['id'] in results['halo_id']) else True
            gal['main'] = True if(ihal['host'] == ihal['id']) else False
    names = gals.dtype.names
    names = [iname for iname in names if(iname[:5] == 'halo_')]
    for gal in gals:
        hal = hals[gal['halo_id']-1]
        gal['dist'] = distance(hal, gal)
        gal['fcontam'] = hal['mcontam']/hal['m']
        if(gal['halo_mvir']>0): continue
        for iname in names:
            gal[iname] = hal[iname[5:]]
    return gals