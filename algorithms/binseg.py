import os
import numpy as np
import pandas as pd
import algorithms.utils as utils
import rpy2.robjects as robjects
import algorithms.utils as utils
import scipy


class binseg: 
    def __init__(self, method, test_stat, max_cp, penalty):
        self.method = method
        self.test_stat = test_stat
        self.max_cp = max_cp
        self.penalty = penalty
        self.penalty_value = 0
        self.ts = None
        self.score = None
        self.cp = None


    def _format_input(self, ts):
        robjects.r("library(changepoint)")
        method_map = {
            "mean": "cpt.mean({})",
            "var": "cpt.var({})",
            "meanvar": "cpt.meanvar({})",
        }
        if self.penalty == "Asymptotic": 
            self.penalty_value = 0.05
        if self.max_cp == "max": 
            self.max_cp = int(len(ts)/10) + 1
        else:
            self.max_cp = 5
        self.method = method_map[self.method]
        self.penalty = "'" + self.penalty + "'"
        self.test_stat = "'" + self.test_stat + "'"
        
        self.ts = robjects.FloatVector(np.array(ts))

    def train(self): 
        pass


    def detect(self, ts):
        self._format_input(ts)
        robjects.r("library(changepoint)")
        robjects.globalenv["mt"] = self.ts
        param_string = "{0}, penalty={1}, test.stat={2}, method='BinSeg', Q={3}, pen.value={4}".format("mt", self.penalty, self.test_stat, self.max_cp, self.penalty_value)
        cmd = self.method.format(param_string)
        robjects.globalenv["mycpt"] = robjects.r(cmd)
        ecp = robjects.r("cpts(mycpt)")
        self.cp = utils.convert_cp_to_intervals(ecp, min_interval_length=2)
        


