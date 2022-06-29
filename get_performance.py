import pandas as pd
import numpy as np
np.random.seed(1234)
import os

from sklearn.metrics import matthews_corrcoef as mcc, confusion_matrix, \
    precision_score, recall_score, accuracy_score, roc_auc_score, plot_confusion_matrix

def bootstrap_metrics(y_true, y_pred, metrics, N=100, k=0.25):
    nsamples = len(y_true)
    
    indices = np.arange(nsamples)
    nboot = int(nsamples*k)
    
    perf = []
    for i in range(N):
        indices_iteration = np.random.choice(indices, nboot)
        y_true_ = y_true[indices_iteration]
        y_pred_ = y_pred[indices_iteration]
        
        perf_iteration = []
        for met in metrics:
            perf_iteration.append(met(y_true_, y_pred_))
        perf.append(np.array(perf_iteration))
    
    perf = np.array(perf)
    # print(perf.shape)
    
    low_CI, median, high_CI = np.apply_along_axis(np.quantile, 0, perf, [0.05, 0.5, 0.95])
    
    return(low_CI, median, high_CI)
    
#%%
DATAROOT = './data'
predictiondir = f'{DATAROOT}/predictions'

files = os.listdir(predictiondir)

files = ['transfer_SilvaNet_ComfTech_Movement___test.csv',
         'transfer_SilvaNet_ComfTech_Movement___train.csv']

#%%
for file in files:
    print(file)
    predictions = pd.read_csv(f'{predictiondir}/{file}', index_col=0)
    #%
    y_true = predictions['target']
    y_pred = predictions['pred']
    #%
    l, med, h = bootstrap_metrics(y_true, y_pred, [mcc])#, precision_score, recall_score])
    #%
    print(l, med, h)

