import os
import numpy as np
import pandas as pd


def data_split(ts, labels, ratio=(0.3,0.3,0.4)): 
    """
    Split the time series and corrisponding cp labels into training, validation, and testing segments.
    Args:
        ts (array):
            Time series data
        labels (list):
            Change point labels
        ratio (tuple): 
            Tuple of taining, validation, and testing in this order (must sum up to 1)
    
    Returns:
        split_ts (dict):
            {
            'train': {ts: array, labels: list},
            'validate': {ts: array, labels: list}, 
            'test': {ts: array, labels: list}
            }

    """
    labels = np.array(labels)
    split_data = {}
    n = len(ts)
    cp = labels[:, 0]
    end_train = int(ratio[0] * n) #exclusive
    end_validate = int((ratio[0]+ratio[1]) * n) #exculsive 
    end_train_labels = np.where(cp < end_train)[0] #exculsive 
    if len(end_train_labels): 
        end_train_labels = end_train_labels[-1] + 1 #exculsive 
    else: 
        end_train_labels = 0
    end_validate_labels = np.where(cp < end_validate)[0]
    if len(end_validate_labels): 
        end_validate_labels = end_validate_labels[-1] + 1 #exculsive 
    else: 
        end_validate_labels = 0
    split_data['train'] = {'ts': ts[:end_train], 'labels': labels[:end_train_labels]}
    split_data['validate'] = {'ts': ts[end_train:end_validate], 'labels': labels[end_train_labels:end_validate_labels] - end_train}
    split_data['test'] = {'ts': ts[end_validate:], 'labels': labels[end_validate_labels:] - end_validate}
    
    return split_data
