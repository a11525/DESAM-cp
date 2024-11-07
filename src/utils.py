import numpy as np
import pandas as pd
import ast 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score , f1_score, recall_score,accuracy_score, average_precision_score
import pdb

def cal_metric(pred_arr, true_arr):
        
    auc = roc_auc_score(true_arr, pred_arr)
    aupr = average_precision_score(true_arr, pred_arr)

    acc = accuracy_score(true_arr, pred_arr.round())
    precision = precision_score(true_arr, pred_arr.round())
    recall = recall_score(true_arr, pred_arr.round())
    f1 = f1_score(true_arr, pred_arr.round())

    return auc,aupr, recall, precision, acc, f1

def multi_class_cal_metric(pred_arr,true_arr):
    
    auc_list =[]
    aupr_list=[]
    class_range = pred_arr.shape[-1]

    for i in range(class_range):
        auc_list.append(roc_auc_score(true_arr[:,i],pred_arr[:,i]))
        aupr_list.append(average_precision_score(true_arr[:,i],pred_arr[:,i]))

    return auc_list, aupr_list