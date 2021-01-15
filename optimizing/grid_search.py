import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import ParameterGrid
from algorithms.utils import save_results_json
from tqdm import tqdm
class grid_search: 
	def __init__(self, params, algorithm, scoring, store_search_results = False, experiment = None, results_path=None): 
		self.params = params
		self.algorithm = algorithm
		self.scoring = scoring
		self.store_search_results = store_search_results
		self.results_path = results_path
		self.experiment = experiment
		self.results_path = results_path
		self.grid = None
		self.best_param = None
		self.ts = None
		self.labels = None
		self.score = []
		self._generate_grid()

	def _generate_grid(self): #todo: make it handel exceptions in parameters combinations
	    self.grid = ParameterGrid(self.params)

	def search(self, ts, labels, margin): 
		self.ts = ts
		self.labels = labels
		for param in tqdm(self.grid): 
                    try: 
                                model = self.algorithm(**param)
                                model.train(self.ts)
                                model.detect(self.ts)
                                score_i = self.scoring(self.labels, model.cp, margin)
                                self.score.append(score_i)
                                print('score:', score_i)
                                if self.store_search_results: 
                                	save_results_json(self.experiment, model, param, score_i, self.results_path)
                    except Exception as error: 
                                print("FAIL", error)
                                self.score.append(-1)
                                if self.store_search_results: 
                                    save_results_json(self.experiment, None, param, None, self.results_path, 'fail', error)
		self.best_param = self.grid[np.argmax(self.score)]

