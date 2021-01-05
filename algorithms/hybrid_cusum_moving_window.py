import os
import numpy as np
import pandas as pd
import algorithms.utils as utils

class hybrid_cusum_moving_window: 

    def __init__(self, window_size, rows, overlap_ratio, rank, singular_threshold, 
           distance_threshold, training_ratio, skip):
        self.rows = rows
        self.cols = int(window_size/rows)
        self.window_size = self.rows*self.cols
        self.rank = rank
        self.singular_threshold = singular_threshold
        self.distance_threshold = distance_threshold
        self.training_ratio = training_ratio
        self.skip = skip
        self.window_overlap = int(self.window_size*overlap_ratio)
        self.ts = None
        self.score = None
        self.cp = None
        self.param_checked, self.param_error_msg = self._check_param()
        if not self.param_checked:
            raise RuntimeError(self.param_error_msg)

    def _check_param(self): 
        ## Rank should not be bigger than matrix dimentsions 
        if self.rank:
            if self.rank > min(self.rows, self.cols) or self.rank <0: 
                return False, 'error in rank parameter'
        # Window size must be at least 4 so the matrix can be 2X2
        if self.window_size < 4:
            return False, 'error in window size'
        ## Minimum size of should be 2
        if self.rows > self.window_size/2 or self.rows < 2: 
            return False, 'error in window rows'
        if self.training_ratio >= 1: 
            return False, 'error in training ratio'
        return True, ''

    def _format_input(self, ts): 
        return np.array(ts) 


    def _estimate_distance_shift_c(self, matrix, r): ## Estimating c using fixed base matrix 
        cols = int(np.shape(matrix)[1]*self.training_ratio)
        if cols < r: 
            raise RuntimeError("training matrix size is less than the rank")
        base_matrix = matrix[:, :cols]
        # Subspace estimation error (eps) 
        U,_,_ = np.linalg.svd(base_matrix, full_matrices=False)
        perp_basis = U[:,r:]
        proj = perp_basis.T @ matrix[:, cols:]
        proj = np.linalg.norm(proj, 2, 0)
        eps = np.max(proj)**2
        # Noise estimation
        U,S,VT = np.linalg.svd(matrix, full_matrices=False)
        noise_matrix = U[:, r:] @ np.diag(S[r:]) @ VT[r:, :]
        number_of_samples = noise_matrix.size
        variance = (1/(number_of_samples-1)) * np.linalg.norm(noise_matrix - np.mean(noise_matrix), 'fro')**2
        return (eps + (self.rows-r)*variance), eps #returns c and eps


    def _estimate_singular_shift_c(self, matrix, r): ## Estimating c using fixed base matrix 
        cols = int(np.shape(matrix)[1]*self.training_ratio)
        if cols < r: 
            raise RuntimeError("training matrix size is less than the rank")
        base_matrix = matrix[:, :cols]
        _,S,_ = np.linalg.svd(base_matrix, full_matrices=False)
        singular_values = S[:r]
        score = np.array([])
        for t in np.arange(cols, np.shape(matrix)[1], 1):
            test_matrix = matrix[:, t-cols:t]
            _,test_singular_values, _ = np.linalg.svd(test_matrix, full_matrices= False)
            test_singular_values = test_singular_values[:r]
            D_t = np.linalg.norm(test_singular_values - singular_values, 2)
            score = np.append(score, D_t)
        return np.max(score)

    def train(self, ts): 
        pass

    def detect(self, ts):
        self.ts = self._format_input(ts)
        if self.ts.ndim == 2: 
            dimentsions = self.ts.shape[1]
        else: 
            dimentsions = 1
        cp = np.zeros_like(self.ts)
        if self.skip: 
            step = self.rows
        else: 
            step = 1
        for d in range(dimentsions): 
           t = 0  
           rebase = True 
           singular_test = True
           if dimentsions == 1: 
                current_ts = self.ts
           else: 
                current_ts = self.ts[:, d]
           singular_score = np.zeros_like(current_ts)
           distance_score = np.zeros_like(current_ts)
           singular_cusum_score = np.zeros_like(current_ts)
           distance_cusum_score = np.zeros_like(current_ts)
           current_cp = np.zeros_like(current_ts)
           while t <= len(current_ts) - self.rows:
                if rebase: 
                    if t > len(current_ts) - self.window_size - self.rows: 
                        print("breaking: not enough data for base matrix")
                        break
                    base_matrix = np.reshape(current_ts[t:t+self.window_size], (self.rows, self.cols), order="F")
                    U,S,ـ = np.linalg.svd(base_matrix, full_matrices= False)
                    if not self.rank: 
                        r = utils.estimate_rank(base_matrix, 0.95)
                    else: 
                        r = self.rank
                    singular_values = S[:r]
                    perp_basis = U[:, r:]
                    singular_shift_c = self._estimate_singular_shift_c(base_matrix, r)
                    distance_shift_c, eps = self._estimate_distance_shift_c(base_matrix, r)
                    singular_h = self.singular_threshold * singular_shift_c
                    distance_h = self.distance_threshold * eps
                    t = t+self.window_size
                    rebase = False
                if t <= len(current_ts) - (self.window_size - self.window_overlap):
                    test_matrix = np.reshape(current_ts[t-self.window_overlap: t + (self.window_size - self.window_overlap)], (self.rows, self.cols), order="F") 
                    _,S, _ = np.linalg.svd(test_matrix, full_matrices= False)
                    test_singular_values = S[:r]
                else: 
                    singular_test = False
                test_vector = current_ts[t:t+self.rows]
                #distance detection 
                D_t = (np.linalg.norm(perp_basis.T @ test_vector, 2))**2 - distance_shift_c
                distance_score[t:t+step] = D_t
                distance_cusum_score[t:t+step] = max(distance_cusum_score[t-1] + D_t, 0)
                if distance_cusum_score[t] >= distance_h:
                    # print("Distance test detection")
                    if self.skip: 
                        current_cp[t] = 1 #Count cp at the begining of the test vector 
                    else: 
                        current_cp[t+self.rows] = 1 #Count cp at the end of the test vector 
                        t = t+self.rows
                    rebase=True 
                    continue 
                #singular values detection 
                if singular_test: 
                    D_t = np.linalg.norm(test_singular_values - singular_values) - singular_shift_c
                    singular_score[t:t+step] = D_t
                    singular_cusum_score[t:t+step] = max(singular_cusum_score[t-1] + D_t, 0)
                    if singular_cusum_score[t] >= singular_h: 
                        # print("singular values test detection")
                        if self.skip: 
                            current_cp[t] = 1 #Count cp at the begining of the test vector 
                        else: 
                            current_cp[t+(self.window_size - self.window_overlap)] = 1 #Count cp at the end of the test vector 
                            t = t+(self.window_size - self.window_overlap)
                        rebase=True 
                        continue
                t = t+step
           cp = cp+current_cp
        cp = 1* (cp!=0)    
        self.cp = utils.convert_binary_to_intervals(cp, min_interval_length=2)
        



