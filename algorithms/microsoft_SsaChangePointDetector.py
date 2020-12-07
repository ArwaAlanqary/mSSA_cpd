import os
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaChangePointDetector
import utils

def _format_input(ts): 
    return pd.Series(ts, name="ts")

def detect(ts, base_window_size, seasonal_window_size, change_history_length, 
              error_function, martingale, power_martingale_epsilon, confidence, 
              columns=None):
    """Microsoft's SSA anomoly detection package.
    http://proceedings.mlr.press/v91/cherubin18a/cherubin18a.pdf
    Args:
        X (list):
            Array containing the time series values.
        base_window_size (int):
            The number of points, N, from the beginning of the sequence used to train the SSA model.
        seasonal_window_size (int): 
            An upper bound, L, on the largest relevant seasonality in the input time-series, which also 
            determines the order of the autoregression of SSA. It must satisfy 2 < L < N/2.
        change_history_length (int): 
            The length of the sliding window on p-value for computing the martingale score.
        error_function (str): 
            The function used to compute the error between the expected and the observed value. 
            Possible values are: {SignedDifference, AbsoluteDifference, SignedProportion, AbsoluteProportion, 
            SquaredDifference}.
        martingale (str): 
            The type of martingale betting function used for computing the martingale score. 
            Available options are {Power, Mixture}.
        power_martingale_epsilon (double): 
            The epsilon parameter for the Power martingale if martingale is set to Power.
        confidence (double): 
            The confidence for change point detection. Possible values are in the range [0, 100].
        columns (????)
    
        Returns:
            (list):
                
    """
    ts = _format_input(ts)
    cpd = SsaChangePointDetector(training_window_size=base_window_size, 
                                 confidence=confidence, 
                                 seasonal_window_size=seasonal_window_size, 
                                 change_history_length=change_history_length, 
                                 error_function=error_function, 
                                 martingale=martingale, 
                                 power_martingale_epsilon=power_martingale_epsilon, 
                                 columns=columns) << {'result': 'ts'}

    cpd.fit(ts, verbose=0)
    output = cpd.transform(ts)
    score = output['result.Martingale Score']
    intervals = utils.convert_binary_to_intervals(output['result.Alert'])

    return score, intervals