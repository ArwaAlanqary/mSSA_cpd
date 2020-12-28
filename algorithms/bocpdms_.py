#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper for BOCPDMS in CPDBench.

Author: G.J.J. van den Burg
Date: 2019-10-02
License: See the LICENSE file.
Copyright: 2019, The Alan Turing Institute

"""

import argparse
import numpy as np
import time

from bocpdms import CpModel, BVARNIG, Detector
from multiprocessing import Process, Manager

from cpdbench_utils import (
    load_dataset,
    make_param_dict,
    exit_with_error,
    exit_with_timeout,
    exit_success,
)

# Ensure overflow errors are raised
# np.seterr(over="raise")

TIMEOUT = 60 * 30  # 30 minutes





def run_bocpdms(mat, params):
    """Set up and run BOCPDMS
    """

    AR_models = []
    for lag in range(params["lower_AR"], params["upper_AR"] + 1):
        AR_models.append(
            BVARNIG(
                prior_a=params["prior_a"],
                prior_b=params["prior_b"],
                S1=params["S1"],
                S2=params["S2"],
                prior_mean_scale=params["prior_mean_scale"],
                prior_var_scale=params["prior_var_scale"],
                intercept_grouping=params["intercept_grouping"],
                nbh_sequence=[0] * lag,
                restriction_sequence=[0] * lag,
                hyperparameter_optimization="online",
            )
        )

    cp_model = CpModel(params["intensity"])

    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(AR_models) for m in AR_models])

    detector = Detector(
        data=mat,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=params["S1"],
        S2=params["S2"],
        T=mat.shape[0],
        store_rl=True,
        store_mrl=True,
        trim_type="keep_K",
        threshold=params["threshold"],
        save_performance_indicators=True,
        generalized_bayes_rld="kullback_leibler",
        # alpha_param_learning="individual",  # not sure if used
        # alpha_param=0.01,  # not sure if used
        # alpha_param_opt_t=30,  # not sure if used
        # alpha_rld_learning=True,  # not sure if used
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss",
    )
    detector.run()

    return detector


def main():
    args = parse_args()

    data, mat = load_dataset(args.input)
    
    detector = run_bocpdms(mat, parameters)
    status = "success"
    
    stop_time = time.time()
    runtime = stop_time - start_time

    
    # According to the Nile unit test, the MAP change points are in
    # detector.CPs[-2], with time indices in the first of the two-element
    # vectors.
    locations = [x[0] for x in detector.CPs[-2]]

    # Based on the fact that time_range in plot_raw_TS of the EvaluationTool
    # starts from 1 and the fact that CP_loc that same function is ensured to
    # be in time_range, we assert that the change point locations are 1-based.
    # We want 0-based, so subtract 1 from each point.
    locations = [loc - 1 for loc in locations]

    # convert to Python ints
    locations = [int(loc) for loc in locations]

    exit_success(data, args, parameters, locations, runtime, __file__)


if __name__ == "__main__":
    main()
