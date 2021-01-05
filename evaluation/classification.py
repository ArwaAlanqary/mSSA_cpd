import os
import numpy as np
import pandas as pd

def _compute_true_positive(actual, detected, margin=0):
    actual_cp = np.array(actual)[:, 0]
    detected_cp = np.array(detected)[:, 0]
    true_positive = 0
    for cp in actual_cp: 
        if min(np.abs(detected_cp - cp)) <= margin: 
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


def compute_f1_score(actual, detected, margin=0):
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