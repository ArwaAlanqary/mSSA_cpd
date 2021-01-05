import os
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaChangePointDetector
import algorithms.utils as utils


class microsoft_ssa: 
  def __init__(self, training_window_size, seasonal_window_size, change_history_length, 
    error_function, martingale, power_martingale_epsilon, confidence):
    self.training_window_size = training_window_size
    self.seasonal_window_size = seasonal_window_size
    self.change_history_length = change_history_length
    self.error_function = error_function
    self.martingale = martingale
    self.power_martingale_epsilon = power_martingale_epsilon
    self.confidence = confidence
    self.ts = None
    self.score = None
    self.cp = None

  def _format_input(self, ts):
    return pd.Series(ts.flatten(), name="ts")

  def train(self, ts): 
    pass

  def detect(self, ts):
    if min(ts.shape) > 1:
        raise
    self.ts = self._format_input(ts)

    cpd = SsaChangePointDetector(training_window_size=self.training_window_size, 
                                 confidence=self.confidence, 
                                 seasonal_window_size=self.seasonal_window_size, 
                                 change_history_length=self.change_history_length, 
                                 error_function=self.error_function, 
                                 martingale=self.martingale, 
                                 power_martingale_epsilon=self.power_martingale_epsilon, 
                                 columns={'result': 'ts'})
    cpd.fit(self.ts, verbose=0)
    output = cpd.transform(self.ts)
    self.score = np.array(output['result.Martingale Score'])
    self.cp = utils.convert_binary_to_intervals(output['result.Alert'])


