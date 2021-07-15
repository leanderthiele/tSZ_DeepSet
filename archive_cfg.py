import os.path

import cfg


def ArchiveCfg() :
    """
    Writes a copy of the -- altered -- cfg module into the results path.
    """
#{{{ 
    with open(os.path.join(cfg.RESULTS_PATH, 'cfg_%s.py'%cfg.ID), 'w') as f :

        for k, v in vars(cfg).items() :

            if k.startswith('__') :
                continue

            f.write(f'{k} = {v}\n')
#}}}
