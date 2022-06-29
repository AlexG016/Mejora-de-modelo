import pyphysio as ph
import os
import numpy as np
import pandas as pd

from config import FSAMP, T_STEP, WINLEN, T, CI_WIN

dataset = 'WCS_FlexComp_baseline'

datadir = f'./data/{dataset}'
traindir = f'./data/dataset/{dataset}/train'
testdir = f'./data/dataset/{dataset}/test'

subjects = np.unique([x.split('.')[0] for x in os.listdir(datadir)])

ID = 0

#%%
part_type = 'train'
outdir = traindir
    
labels = pd.DataFrame(columns = ['label', 'sub', 't_start', 't_beat', 'dataset', 'part_type'])

for sub in subjects[:12]:
    print(sub)
 
    ecg = ph.nature2type(ph.from_pickle(f'{datadir}/{sub}/baseline/ecg_FC_b.pkl'))
    ibi = ph.nature2type(ph.from_pickle(f'{datadir}/{sub}/baseline/ibi_FC_b.pkl'))
    
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
            
            ecg_portion = ecg.segment_time(t_start, t_start + WINLEN+1).resample(FSAMP)
            
            
            labels.loc[ID,:] = [is_beat, sub, t_start, t_beat, dataset, part_type]
            
            np.save(f'{outdir}/data/{dataset}_{ID}', ecg_portion.get_values()[0:250])
            
            ID+=1

labels.to_csv(f'{outdir}/labels.csv')

#%%
#ID = 14748
part_type = 'test'
outdir = testdir
    
labels = pd.DataFrame(columns = ['label', 'sub', 't_start', 't_beat', 'dataset', 'part_type'])

for sub in subjects[12:]:
    print(sub)
    
    ecg = ph.nature2type(ph.from_pickle(f'{datadir}/{sub}/baseline/ecg_FC_b.pkl'))
    ibi = ph.nature2type(ph.from_pickle(f'{datadir}/{sub}/baseline/ibi_FC_b.pkl'))
    
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
            
            ecg_portion = ecg.segment_time(t_start, t_start + WINLEN+1).resample(FSAMP)
            
            labels.loc[ID,:] = [is_beat, sub, t_start, t_beat, dataset, part_type]
            
            np.save(f'{outdir}/data/{dataset}_{ID}', ecg_portion.get_values()[:250])
            
            ID+=1

labels.to_csv(f'{outdir}/labels.csv')
