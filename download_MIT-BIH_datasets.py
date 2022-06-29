import os
import numpy as np

import wfdb

#%%
print('NOTE: Downloaded datasets are subject to the  Open Data Commons Attribution License (v1).')
print('Copy of the license is available in the ODC_license_v1.txt file')

OUTDIR = './data/original/MIT_BIH_Arrhythmia'
wfdb.dl_database('mitdb', dl_dir=OUTDIR)

OUTDIR = './data/original/MIT_BIH_LongTerm'
wfdb.dl_database('ltdb', dl_dir=OUTDIR)

OUTDIR = './data/original/MIT_BIH_NormalSinus'
wfdb.dl_database('nsrdb', dl_dir=OUTDIR)

