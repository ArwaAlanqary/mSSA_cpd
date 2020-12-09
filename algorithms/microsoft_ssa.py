import os
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaChangePointDetector
import utils

class microsoft_ssa: 
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
	def __init__(self, base_window_size, seasonal_window_size, change_history_length, 
              error_function, martingale, power_martingale_epsilon, confidence, 
              columns):
		self.base_window_size = base_window_size
		self.seasonal_window_size = seasonal_window_size
		self.change_history_length = change_history_length
		self.error_function = error_function
		self.martingale = martingale
		self.power_martingale_epsilon = power_martingale_epsilon
		self.confidence = confidence
		self.columns = columns
		self.ts = None
		self.score = None
		self.cp = None

	def _format_input(self, ts):
		return pd.Series(ts, name="ts")


	def train(self): 
		pass

	def detect(self, ts):
		cpd = SsaChangePointDetector(training_window_size=self.base_window_size, 
                                 confidence=self.confidence, 
                                 seasonal_window_size=self.seasonal_window_size, 
                                 change_history_length=self.change_history_length, 
                                 error_function=self.error_function, 
                                 martingale=self.martingale, 
                                 power_martingale_epsilon=self.power_martingale_epsilon, 
                                 columns=self.columns) << {'result': 'ts'}
		ts = self._format_input(self.ts)
		cpd.fit(ts, verbose=0)
		output = cpd.transform(ts)
		self.score = np.array(output['result.Martingale Score'])
		self.cp = utils.convert_binary_to_intervals(output['result.Alert'])


