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

    try :
        delta_t = ( d['datetime_complete']-d['datetime_start'] ).sum()
    except (KeyError, TypeError) :
        print('%s not yet populated'%fname)
        continue
    
    if total_time is None :
        total_time = delta_t
    else :
        total_time += delta_t

    print('%s :'%fname)
    print('\t%s'%str(delta_t.round('s')))

print('-------------------------------------')
print('%s total'%str(total_time.round('s')))
print('= %.0f hours'%(total_time.total_seconds()/60/60))
