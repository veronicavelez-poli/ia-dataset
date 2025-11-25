import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    
    tp = ((y_true==1)&(y_pred==1)).sum()
    tn = ((y_true==0)&(y_pred==0)).sum()
    fp = ((y_true==0)&(y_pred==1)).sum()
    fn = ((y_true==1)&(y_pred==0)).sum()
    
    return tn, fp, fn, tp

def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return (tp+tn)/(tp+tn+fp+fn)

def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tp/(tp+fp + 1e-8)

def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tp/(tp+fn + 1e-8)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r/(p+r+1e-8)