import os
import numpy as np
import pandas as pd
import algorithms.utils as utils
import rpy2.robjects as robjects
import algorithms.utils as utils
import scipy


class binseg: 
    def __init__(self, method , max_cp, test_stat, penalty , penalty_value = 0):
        self.method = method
        self.test_stat = test_stat
        self.max_cp = max_cp
        self.penalty = penalty
        self.penalty_value = penalty_value


    def _format_input(self, ts):
        robjects.r("library(changepoint)")
        method_map = {
            "mean": "cpt.mean({})",
            "var": "cpt.var({})",
            "meanvar": "cpt.meanvar({})",
        }
        self.method = method_map[self.method]
        self.penalty = "'" + self.penalty + "'"
        self.test_stat = "'" + self.test_stat + "'"
        self.ts = robjects.FloatVector(np.array(ts))

    def train(self, ts): 
        pass


    def detect(self, ts):
        self._format_input(ts)
        robjects.r("library(changepoint)")
        robjects.globalenv["mt"] = self.ts
        param_string = "{0}, penalty={1}, test.stat={2}, method='BinSeg', Q={3}, pen.value={4}".format("mt", self.penalty, self.test_stat, self.max_cp, self.penalty_value)
        cmd = self.method.format(param_string)
        robjects.globalenv["mycpt"] = robjects.r(cmd)
        ecp = robjects.r("cpts(mycpt)")
        self.cp = utils.convert_cp_to_intervals(ecp, len(ts), min_interval_length=2)
        


