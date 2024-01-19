import os
import numpy as np
import pandas as pd

def _compute_true_positive(actual, detected, margin=0):
    actual_cp = np.array(actual)[:, 0]
    detected_cp = np.array(detected)[:, 0]
    true_positive = 0
    for cp in actual_cp: 
        arg_min = np.argmin(np.abs(detected_cp - cp))
        val_min = min(np.abs(detected_cp - cp))
        if val_min <= margin: 
            detected_cp[arg_min] = np.array(actual)[-1, 1] + margin*10
            true_positive += 1
    return true_positive


def _compute_recall(actual, detected, margin=0):
    actual_cp_number = len(actual)
    true_positive = _compute_true_positive(actual, detected, margin)
    return true_positive/actual_cp_number


def _compute_precision(actual, detected, margin=0):
    total_number_of_detected_cp = len(detected)
    true_positive = _compute_true_positive(actual, detected, margin)
    return true_positive/total_number_of_detected_cp


def f1_score(actual, detected, margin=0):
    """
    Compute the the f1 score.
    Args:
        actual (list of tuples):
            Ground truth intervals.
        observed (list of tuples):
            Detected intervals.
        margin (int):
            The margin of error allowed in the time of change detection.
            
    Returns:
        f1_score (float):
            The f1 score 
    """
    recall = _compute_recall(actual, detected, margin)
    precision = _compute_precision(actual, detected, margin)
    return 2*recall*precision/(recall+precision)