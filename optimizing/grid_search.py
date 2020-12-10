import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

class grid_search: 
	def __init__(self, param, algorithm, scoring): 
		self.param = param
		self.algorithm = algorithm
		self.scoring = scoring
		self.grid = None
		self.best_param = None
		self.ts = None
		self.labels = None
		self.score = []
		self._generate_grid()

	def _generate_grid(self): #todo: make it handel exceptions in parameters combinations
	    self.grid = ParameterGrid(self.param)

	def search(self, ts, labels, margin): 
		self.ts = ts
		self.labels = labels
		for params in self.grid: 
			model = self.algorithm(**params)
			model.train()
			model.detect(self.ts)
			self.score.append(self.scoring(self.labels, model.cp, margin))
		self.best_param = self.grid[np.argmax(self.score)]



	    
    

