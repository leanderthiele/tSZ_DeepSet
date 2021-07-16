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

            # do not print the internal variables
            if k.toupper() != k :
                continue

            # do not print the constant variables
            if k.startswith('_') :
                continue

            # we want strings in the output to be quoted
            if isinstance(v, str) :
                v = '"%s"'%v

            f.write(f'{k} = {v}\n')
#}}}
