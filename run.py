import os
import sys
import pandas as pd
from config import DATADIR, DATASETS, ALGORITHMS, PARAMS, METRICS, MARGIN, RATIO

from optimizing.grid_search import grid_search
from evaluation.classification import compute_f1_score

from algorithms.utils import data_split, save_results_json, save_results_table

#Specify experiment: 
algorithm_name = 'binseg'
dataset = 'synth'
data_names = ['mean'] ##All data files in the dataset
metric = 'compute_f1_score'

#Specify paths
search_results_path = os.path.join(os.getcwd(), 'results', 'search', algorithm_name, dataset)
test_restuls_path = os.path.join(os.getcwd(), 'results', 'test', algorithm_name, dataset)
data_path = os.path.join(os.getcwd(), DATADIR)

for data_name in data_names:
	##Prepare experiment and paths
	experiment = {'dataset': dataset, 
				'data_name': data_name, 
				'algorithm_name': algorithm_name, 
				'metric': metric}
	##Load data
	data = pd.read_csv(os.path.join(data_path,  dataset,"{}_ts.csv".format(data_name)), header=None)
	labels = pd.read_csv(os.path.join(data_path, dataset,"{}_labels.csv".format(data_name)), header=None)
	ts = data[1].values
	splitted_data = data_split(ts, labels, RATIO)
	##Search for best parameters
	optimizer = grid_search(PARAMS[algorithm_name], ALGORITHMS[algorithm_name], METRICS[metric], True, experiment, search_results_path)
	optimizer.search(ts, labels, MARGIN)
	## Generate model results with best parameteres
	try: 
		model = ALGORITHMS[algorithm_name](**optimizer.best_param)
		model.train()
		model.detect(ts)
		score = METRICS[metric](labels, model.cp, MARGIN)
		save_results_json(experiment, model, optimizer.best_param, score, test_restuls_path)
		save_results_table(experiment, score, test_restuls_path)
	except Exception as error:
		print(error)
		save_results_json(experiment, None, optimizer.best_param, None, test_restuls_path, status='fail', error = error)
		save_results_table(experiment, None, test_restuls_path, status='fail')


