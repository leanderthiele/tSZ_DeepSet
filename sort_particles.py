import subprocess
import shutil
import os.path

import numpy as np

import cfg

idx = 0

while True :

    print(idx)
    
    try :
        globals_DM = eval(open(cfg._STORAGE_FILES['DM']%(idx, 'globals'), 'r').readline())
    except FileNotFoundError :
        print('Found %d halos'%idx)
        break

    # make safety copy of data if it does not already exist
    for s in ['coords', 'velocities', ] :
        if not os.path.isfile(cfg._STORAGE_FILES['DM']%(idx, '%s_backup'%s)) :
            shutil.copyfile(cfg._STORAGE_FILES['DM']%(idx, s),
                            cfg._STORAGE_FILES['DM']%(idx, '%s_backup'%s))

    # get the geometry
    R200c = globals_DM['R200c']
    pos = globals_DM['pos']
    ul_corner = pos - 2.51 * R200c
    extent = 2 * 2.51 * R200c

    # now call the compiled code
    subprocess.run(['./sort_particles', str(idx), *[str(x) for x in ul_corner], str(extent)], check=True)

    idx += 1
