# ...
# check names of dirs
# ...

import pyphysio as ph
import os
import numpy as np
import pandas as pd
import wfdb

from config import FSAMP, T_STEP, WINLEN, T, CI_WIN

dataset = 'MIT_BIH_LongTerm'

datadir = f'./data/{dataset}'
traindir = f'./data/dataset/{dataset}/train'
testdir = f'./data/dataset/{dataset}/test'

subjects = np.unique([x.split('.')[0] for x in os.listdir(datadir)])

ID = 0

#%%
part_type = 'train'
outdir = traindir
    
labels = pd.DataFrame(columns = ['label', 'sub', 't_start', 't_beat', 'dataset', 'part_type'])

for sub in subjects[:5]:
    print(sub)

    signal, fields = wfdb.rdsamp(f'{datadir}/{sub}') 
    annotation = wfdb.rdann(f'{datadir}/{sub}', 'atr')
    beats_idx = annotation.sample
    fsamp = fields['fs']
    
    ibi_values = np.diff(beats_idx)/fsamp
    
    ecg = ph.EvenlySignal(signal[:,0], fsamp)
    ibi = ph.UnevenlySignal(ibi_values, fsamp, x_values = beats_idx[1:], x_type='indices')
    
    ibi_times = ibi.get_times()
    
    fsamp = ecg.get_sampling_freq
    t_st = ecg.get_start_time()
    t_sp = ecg.get_end_time() - 2* WINLEN
    if (t_sp - t_st) > 3600:
        t_sp = t_st + 3600
        
    for t_start in np.arange(t_st, t_sp, T_STEP):
        idx_beats_after = np.where((ibi_times - t_start) > 0)[0]
        if len(idx_beats_after)> 0:
            t_beat = ibi_times[idx_beats_after[0]]
            
            #check if portion contains a beat
            is_beat = 0
            
            if ((t_beat - t_start) >= (T - CI_WIN/2) and \
                (t_beat - t_start) <= (T + CI_WIN/2)):
                is_beat = 1
            
            ecg_portion = ecg.segment_time(t_start, t_start + WINLEN).resample(FSAMP)
            
            labels.loc[ID,:] = [is_beat, sub, t_start, t_beat, dataset, part_type]
            
            np.save(f'{outdir}/data/{dataset}_{ID}', ecg_portion.get_values())
            
            ID+=1

labels.to_csv(f'{outdir}/labels.csv')

#%%
# ID = 72000
part_type = 'test'
outdir = testdir
    
labels = pd.DataFrame(columns = ['label', 'sub', 't_start', 't_beat', 'dataset', 'part_type'])

for sub in subjects[5:]:
    print(sub)
    
    signal, fields = wfdb.rdsamp(f'{datadir}/{sub}') 
    annotation = wfdb.rdann(f'{datadir}/{sub}', 'atr')
    beats_idx = annotation.sample
    fsamp = fields['fs']
    
    ibi_values = np.diff(beats_idx)/fsamp
    
    ecg = ph.EvenlySignal(signal[:,0], fsamp)
    ibi = ph.UnevenlySignal(ibi_values, fsamp, x_values = beats_idx[1:], x_type='indices')
    
    ibi_times = ibi.get_times()
    
    fsamp = ecg.get_sampling_freq
    t_st = ecg.get_start_time()
    t_sp = ecg.get_end_time() - 2* WINLEN
    if (t_sp - t_st) > 3600:
        t_sp = t_st + 3600
        
    for t_start in np.arange(t_st, t_sp, T_STEP):
        idx_beats_after = np.where((ibi_times - t_start) > 0)[0]
        if len(idx_beats_after)> 0:
            t_beat = ibi_times[idx_beats_after[0]]
            
            #check if portion contains a beat
            is_beat = 0
            
            if ((t_beat - t_start) >= (T - CI_WIN/2) and \
                (t_beat - t_start) <= (T + CI_WIN/2)):
                is_beat = 1
            
            ecg_portion = ecg.segment_time(t_start, t_start + WINLEN).resample(FSAMP)
            
            labels.loc[ID,:] = [is_beat, sub, t_start, t_beat, dataset, part_type]
            
            np.save(f'{outdir}/data/{dataset}_{ID}', ecg_portion.get_values())
            
            ID+=1

labels.to_csv(f'{outdir}/labels.csv')
