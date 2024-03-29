import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
import uuid

def format_params(algorithm, params, meta_data): 
    if algorithm == "binseg":
        if params["max_cp"] == "max": 
            params["max_cp"] = meta_data["num_cp"]
        else:
            params["max_cp"] = meta_data["num_cp"] / meta_data["num_ts"]
        if params["penalty"] == "Asymptotic": 
            params["penalty_value"] = 0.05
        else: 
            params["penalty_value"] = 0
        return params

    elif algorithm == "mssa" or algorithm == "mssa_mw": 
        params["window_size"] = int(params["window_size"] * meta_data["mean_interval"])
        params["rows"] = int(np.sqrt(params["rows"]* params["window_size"]))
        return params

    elif algorithm == "microsoft_ssa": 
        params["training_window_size"] = int(params["training_window_size"] * meta_data["mean_interval"])
        params["seasonal_window_size"] = int(np.sqrt(params["seasonal_window_size"]* params["training_window_size"]))
        return params 

    else: 
        return params 


def convert_binary_to_intervals(binary_array, min_interval_length=2): 
    """
    Converting a binary array of change points into a list of intervals.
    Args:
        binary_array (list):
            Binary array with the same length of the timeseries.
        min_interval_length (int):
            Minimum allowed length of interval; min spacing between condecutive changepoints.
    
    Returns:
        list_of_intervals (list):
            List of tuples indicating the start and and of an interval.                 
    """

    list_of_intervals = list()
    binary_array = np.array(binary_array)
    change_points = np.nonzero(binary_array)[0]
    if len(change_points) == 0: 
        list_of_intervals.append([0, len(binary_array)-1])
        return list_of_intervals
    if change_points[0] != 0: 
        change_points = np.insert(change_points,0,0)
    start = change_points[0]
    i = start
    if min_interval_length: 
        while i < len(change_points)-1:
            if change_points[i+1] - start < min_interval_length:
                i += 1
                continue 
            list_of_intervals.append([start, change_points[i+1]-1])
            
            start = change_points[i+1]
            i += 1
        if len(binary_array) - 1 - list_of_intervals[-1][-1] < min_interval_length: 
            list_of_intervals[-1][-1] = len(binary_array) - 1
        else:
            list_of_intervals.append([start, len(binary_array) - 1])
    
    return list_of_intervals  


def convert_cp_to_intervals(change_points, len_ts, min_interval_length=2): 
    """
    Converting a binary array of change points into a list of intervals.
    Args:
        binary_array (list):
            Binary array with the same length of the timeseries.
        min_interval_length (int):
            Minimum allowed length of interval; min spacing between condecutive changepoints.
    
    Returns:
        list_of_intervals (list):
            List of tuples indicating the start and and of an interval.                 
    """

    list_of_intervals = list()
    change_points = np.sort(np.array(change_points))
    if len(change_points) == 0: 
        list_of_intervals.append([0, len_ts-1])
        return list_of_intervals
    if change_points[0] != 0: 
        change_points = np.insert(change_points,0,0)
    start = int(change_points[0])
    i = start
    if min_interval_length: 
        while i < len(change_points)-1:
            if change_points[i+1] - start < min_interval_length:
                i += 1
                continue 
            list_of_intervals.append([start, change_points[i+1]-1])
            
            start = change_points[i+1]
            i += 1
    list_of_intervals.append([change_points[-1], len_ts-1])
    return list_of_intervals  


def estimate_rank(mat, th = 0.9):
    _,S,_ = np.linalg.svd(mat)
    # S = S**2
    S = np.cumsum(S)
    threshold = th*S[-1]
    r = np.argmax(S>threshold)
    return r+1


def scale_ts(x, method='minmax'):
    x = np.array(x)
    if method == 'normalize': 
        return (x - np.mean(x))/ np.std(x)
    if method == 'minmax': 
        return (x - np.min(x))/ (np.max(x) - np.min(x))


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

def json_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

def save_results_json(experiment, model, param, score, path, status='success', error=''): 
    results = {}
    results['status'] = status
    results['error'] = str(error)
    results['algorithm'] = experiment['algorithm_name']
    results['dataset'] = experiment['dataset']
    results['data_name'] = experiment['data_name']
    results['param'] =  param
    if status == 'success':
        results['cp'] = model.cp
        results['score'] = {'metric': experiment['metric'], 'value': score}
    else: 
        results['cp'] = None
        results['score'] = None
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = experiment['data_name'] + "_" + _generate_file_uuid(param)
    with open('{0}/{1}.json'.format(path, file_name), 'w') as file:
        json.dump(results, file, default=json_converter)

def save_results_table(experiment, score, path, status='success'): 
# store algorithm/data/f1_score 
    with open('{}/results_table.csv'.format(path),'a+') as outfile:
        algorithm_name = experiment['algorithm_name']
        dataset = experiment['dataset']
        data_name = experiment['data_name']
        if status == 'success': 
            outfile.write(f'{algorithm_name},{dataset}, {data_name}, {status}, {score},\n')
        else: 
            outfile.write(f'{algorithm_name},{dataset}, {data_name},{status},{None},\n')

def _generate_file_uuid(param):
    string_name = "".join(map(str,param.values()))
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, string_name))








