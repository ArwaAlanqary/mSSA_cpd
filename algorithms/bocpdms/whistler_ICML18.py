#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:00:45 2018

@author: jeremiasknoblauch

Description: Reads in and processes Whistler data
             Note: for 0-periods, get NIG model with high confidence of 0s
"""

import numpy as np
import csv, os
import matplotlib.pyplot as plt

from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel


baseline_working_directory = os.getcwd()
baseline_working_directory = baseline_working_directory.replace("/", "//") 
baseline_working_directory = baseline_working_directory + "//Data"
whistler_data = baseline_working_directory + "//whistler_data.csv"
whistler_dates = baseline_working_directory + "//whistler_dates.csv"


"""STEP 0: Decide what to do"""
run_detectors = True
save_plots = False
print_plots = False
normalize = True
log_transform=True

"""STEP 1: Read in and convert to float"""
raw_data = []
count = 0 
with open(whistler_data) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count>0:
            raw_data += row
        count += 1

raw_data_float = []
for entry in raw_data:
    raw_data_float.append(float(entry))
raw_data = raw_data_float

"""STEP 2: put into right form"""
T = int(len(raw_data))
S1, S2 = 1,1
data = np.array(raw_data).reshape(T,1)
mean, variance = np.mean(data[np.where(data>0.0)]), np.var(data[np.where(data>0.0)])



"""STEP 4: Saatci et al. log-transform the data"""
if log_transform:
    data = np.log(data+1)
    
"""STEP 3: Saatci et al. normalize the data"""
if normalize:
    data = (data-np.mean(data))/np.sqrt(np.var(data))
    
minimum = np.min(data)
mean = np.mean(data[np.where(data>minimum)])
variance = np.var(data[np.where(data>minimum)])


"""STEP 5: Set up cp model and lag lengths"""
intensity = 100 
cp_model = CpModel(intensity)
mult = 1.5
upper_AR = int(mult*pow(float(T)/np.log(T), 0.25) + 1)
lower_AR = 1

"""STEP 6: Decide on priors & activate hyperparameter opt."""
prior_a = 1
prior_b = 1
intercept_scale = 0 #Corresponds to saying that our prior belief is that
                          #no snow falls. (because minimum is like zero before
                          #the normalized log transform)
prior_var_scale = 1
hyperpar_opt = "caron"

"""Set up models"""
AR_models = []

#fit only constant
AR_models += [BVARNIG(prior_a = prior_a,prior_b = prior_b,
            S1 = S1,S2 = S2,
            prior_mean_beta = minimum, #i.e., set prior to 0 snowfall
            prior_var_scale = prior_var_scale,
            intercept_grouping = None,
            nbh_sequence = None,
            restriction_sequence = None,
            hyperparameter_optimization = hyperpar_opt)]

for lag in range(lower_AR, upper_AR+1):
    
    #Set the prior of the coefficients s.t. the intercept can be different from
    #the lag-coefficient priors.
    prior_mean_beta = np.array([intercept_scale] + [0]*lag) 
    
    
    #fit some dynamic parts
    AR_models += [BVARNIG(
                    prior_a = prior_a,prior_b = prior_b,
                    S1 = S1,S2 = S2,
                    prior_mean_beta = prior_mean_beta,
                    prior_var_scale = prior_var_scale,
                    intercept_grouping = None,
                    nbh_sequence = [0]*lag,
                    restriction_sequence = [0]*lag,
                    hyperparameter_optimization = hyperpar_opt)]
    

"""model universe and model priors"""
model_universe = np.array(AR_models)
model_prior = np.array([1/len(model_universe)]*len(model_universe))

"""run detectors, potentially plot stuff"""
"""Build and run detector"""
detector = Detector(data=data, 
        model_universe=model_universe, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, S2 = S2, T = T, 
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold = 200,
        training_period = 25, #i.e., we let 2 years pass before MSE computed
        notifications = 1500,
        save_performance_indicators = True)
detector.run()

"""Store results + real CPs into EvaluationTool obj"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_run_detector(detector)
        

"""plot transformed data"""
fig = EvT.plot_raw_TS(data.reshape(T,1))
"""plot prediction error"""
fig = EvT.plot_prediction_error(data, indices=[0], print_plt=True,
                                time_range = np.linspace(2*365,
                             T-upper_AR-1,
                             T-upper_AR-1-2*365,
                             dtype=int))
"""plot predictions themselves"""
fig = EvT.plot_predictions(
        indices = [0], print_plt = True, 
        legend = False, 
        legend_labels = None, 
        legend_position = None, 
        time_range = None,
        show_var = False, 
        show_CPs = True)

plt.show()

print("************PRED MSE + NLL AS IN PAPER************")
print("MSE", np.sum(np.mean(detector.MSE, axis=0)))
print("NLL", np.mean(detector.negative_log_likelihood))
print("**************************************************")













