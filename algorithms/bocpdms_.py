import os
import numpy as np
import pandas as pd
import algorithms.utils

from algorithms.bocpdms import CpModel, BVARNIG, Detector

class BOCPDMS: 
  def __init__(self, prior_a = 0.1, prior_b = 0.1, prior_mean_scale = 0, prior_var_scale= 1, intercept_grouping= None,intensity = 50,threshold = 100):
    self.prior_a = prior_a
    self.prior_b = prior_b
    self.prior_mean_scale = prior_mean_scale
    self.prior_var_scale = prior_var_scale
    self.intercept_grouping = intercept_grouping
    self.intensity = intensity
    self.threshold = threshold

  def _format_input(self, ts):
    
    if len(ts.shape) == 1:
        ts = ts.reshape(-1,1) 

    self.S1 = ts.shape[1]
    # I beleive this is meant for order-3 tensors data?
    self.S2 = 1
    self.T = ts.shape[0]
    Lmin = 1
    Lmax = int(pow(self.T / np.log(self.T), 0.25) + 1)
    self.lower_AR = Lmin
    self.upper_AR = Lmax
    
    return ts

  def train(self): 
    pass

  def detect(self, ts):
    mat = self._format_input(ts)
    
    self.detector = self._run_bocpdms(ts)
    locations = [x[0] for x in self.detector.CPs[-2]]

    # Based on the fact that time_range in plot_raw_TS of the EvaluationTool
    # starts from 1 and the fact that CP_loc that same function is ensured to
    # be in time_range, we assert that the change point locations are 1-based.
    # We want 0-based, so subtract 1 from each point.
    locations = [loc - 1 for loc in locations]

    # convert to Python ints
    locations = [int(loc) for loc in locations]
    cp = np.zeros(self.T)

    cp[locations] = 1
    cp[0] = 0
    self.cp = utils.convert_binary_to_intervals(cp)
   

  def _run_bocpdms(self, mat):
    """
    Set up and run BOCPDMS
    """

    AR_models = []
    for lag in range(self.lower_AR, self.upper_AR + 1):
        AR_models.append(
            BVARNIG(
                prior_a=self.prior_a,
                prior_b=self.prior_b,
                S1=self.S1,
                S2=self.S2,
                prior_mean_scale= self.prior_mean_scale,
                prior_var_scale= self.prior_var_scale,
                intercept_grouping= self.intercept_grouping,
                nbh_sequence=[0] * lag,
                restriction_sequence=[0] * lag,
                hyperparameter_optimization="online",
            )
        )

    cp_model = CpModel(self.intensity)

    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(AR_models) for m in AR_models])

    detector = Detector(
        data=mat,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=self.S1,
        S2=self.S2,
        T=mat.shape[0],
        store_rl=True,
        store_mrl=True,
        trim_type="keep_K",
        threshold=self.threshold,
        save_performance_indicators=True,
        generalized_bayes_rld="kullback_leibler",
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss",
    )
    detector.run()

    return detector
