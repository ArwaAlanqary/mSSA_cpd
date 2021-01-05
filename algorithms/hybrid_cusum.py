import os
import numpy as np
import pandas as pd
import algorithms.utils as utils


class hybrid_cusum: 

    def __init__(self, window_size, rows, rank, singular_threshold, 
           distance_threshold, training_ratio, skip):
        self.rows = rows
        self.cols = int(window_size/rows)
        self.window_size = self.rows*self.cols
        self.rank = rank
        self.singular_threshold = singular_threshold
        self.distance_threshold = distance_threshold
        self.training_ratio = training_ratio
        self.skip = skip
        self.ts = None
        self.score = None
        self.cp = None

    def _check_param(self): 
        ## Rank should not be bigger than matrix dimentsions 
        if self.rank:
            if self.rank > min(self.rows, self.cols) or self.rank <= 0: 
                raise RuntimeError('Rank is bigger then matrix size')
        # Window size must be at least 4 so the matrix can be 2X2
        if self.window_size < 4:
            raise RuntimeError('Window size is too small, must be > 4')
        ## Minimum size of should be 2
        if self.rows > self.window_size/2 or self.rows < 2: 
            raise RuntimeError('Number of rows is not correct, must be 2 < rows <window_size/2')
        if self.training_ratio >= 1: 
            raise RuntimeError('Training ratio must be less than 1')

    def _format_input(self, ts): 
        return np.array(ts) 

    def _estimate_distance_shift_c(self, matrix, r): 
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

        return (eps + (self.rows-r)*variance), eps


    def _estimate_singular_shift_c(self, matrix,r): 
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
            cp = np.zeros_like(self.ts[:, 0])
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
            if dimentsions == 1: 
                current_ts = self.ts
            else: 
                current_ts = self.ts[:, d]
            singular_score = np.zeros_like(current_ts)
            distance_score = np.zeros_like(current_ts)
            singular_cusum_score = np.zeros_like(current_ts)
            distance_cusum_score = np.zeros_like(current_ts)
            current_cp = np.zeros_like(current_ts)
            while t <= len(current_ts)-step:
                if rebase: 
                    if t > len(current_ts) - step - self.window_size: 
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
                    t += self.window_size
                    rebase = False
                test_matrix = np.reshape([current_ts[t-self.rows*self.cols:t]], (self.rows, self.cols), order="F") 
                test_vector = test_matrix[:, -1]
                _,S, _ = np.linalg.svd(test_matrix, full_matrices= False)
                test_singular_values = S[:r]
                #distance detection 
                D_t = (np.linalg.norm(perp_basis.T @ test_vector, 2))**2 - distance_shift_c
                distance_score[t:t+step] = D_t
                distance_cusum_score[t:t+step] = max(distance_cusum_score[t-1] + D_t, 0)
                #singular values detection 
                D_t = np.linalg.norm(test_singular_values - singular_values) - singular_shift_c
                singular_score[t:t+step] = D_t
                singular_cusum_score[t:t+step] = max(singular_cusum_score[t-1] + D_t, 0)
                
                if distance_cusum_score[t] >= distance_h:
                    current_cp[t] = 1
                    rebase=True 
                    continue
                if singular_cusum_score[t] >= singular_h: 
                    current_cp[t] = 1
                    rebase=True 
                    continue
                t = t+step
            cp = cp+current_cp
        cp = 1* (cp!=0)
        self.cp = utils.convert_binary_to_intervals(cp, min_interval_length=2)


