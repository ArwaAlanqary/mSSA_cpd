import os
import numpy as np
import pandas as pd
import algorithms.utils as utils
from sklearn.preprocessing import StandardScaler

class no_change: 

    def __init__(self):
        self.cp = None

    def train(self, ts): 
        pass

    def detect(self, ts):
        self.cp = [[0, len(ts)-1]]
        
        



