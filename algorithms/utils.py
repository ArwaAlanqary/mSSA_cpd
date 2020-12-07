import os
import numpy as np
import pandas as pd

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


