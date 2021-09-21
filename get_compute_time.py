"""
small script to get the total time in (GPU+9CPU)*hrs expended on Optuna
hyperparameter searches
"""

from glob import glob

import optuna


fnames = glob('*.db')

total_time = None

for fname in fnames :
    study = optuna.load_study(study_name=fname[:-3], storage='sqlite:///%s'%fname)
    d = study.trials_dataframe()
    delta_t = ( d['datetime_complete']-d['datetime_start'] ).sum()
    
    if total_time is None :
        total_time = delta_t
    else :
        total_time += delta_t

    print('%s for %s'%(str(delta_t.round('s')), fname))

print('-------------------------------------')
print('%s total'%str(total_time.round('s')))
