import os
import numpy as np
import pandas as pd
import algorithms.utils as utils


class hybrid_cusum: 

    def __init__(self, window_size, rows, rank, singular_threshold, 
           distance_threshold, training_ratio, skip):
        self.window_size = window_size
        self.rows = rows
        self.rank = rank
        self.singular_threshold = singular_threshold
        self.distance_threshold = distance_threshold
        self.training_ratio = training_ratio
        self.skip = skip
        self.ts = None
        self.score = None
        self.cp = None


    def _estimate_distance_shift_c(self, matrix): 
        base_cols = int(np.shape(matrix)[1]*self.training_ratio)
        rows = np.shape(matrix)[0]
        base_matrix = matrix[:, :base_cols]
        U,_,_ = np.linalg.svd(base_matrix, full_matrices=False)
        perp_basis = U[:,self.rank:]
        proj = perp_basis.T @ matrix[:, base_cols:]

        U,S,VT = np.linalg.svd(matrix, full_matrices=False)
        noise_matrix = U[:, self.rank:] @ np.diag(S[self.rank:]) @ VT[self.rank:, :]
        number_of_samples = np.shape(noise_matrix)
        number_of_samples = number_of_samples[0] * number_of_samples[1]
        proj = np.linalg.norm(proj, 2, 0)
        variance = (1/(number_of_samples-1)) * np.linalg.norm(noise_matrix - np.mean(noise_matrix), 'fro')**2
        eps = np.max(proj)**2

        return (eps + (rows-self.rank)*variance), eps


    def _estimate_singular_shift_c(self, matrix): 
        cols = int(np.shape(matrix)[1]*self.training_ratio)
        if cols < self.rank: 
            print("matrix size is less than the rank")
        base_matrix = matrix[:, :cols]
        _,S,_ = np.linalg.svd(base_matrix, full_matrices=False)
        S = S[:self.rank]
        score = np.array([])
        for t in np.arange(cols, np.shape(matrix)[1], 1):
            test_matrix = matrix[:, t-cols:t]
            _,test_singular_values, _ = np.linalg.svd(test_matrix, full_matrices= False)
            test_singular_values = test_singular_values[:self.rank]
            D_t = np.linalg.norm(test_singular_values - S)
            score = np.append(score, D_t)
        return np.max(score)

    def train(self, ts): 
        pass

    def detect(self, ts):
        self.ts = ts
        cols = int(self.window_size/self.rows)
        singular_score = np.zeros_like(ts)
        distance_score = np.zeros_like(ts)
        singular_cusum_score = np.zeros_like(ts)
        distance_cusum_score = np.zeros_like(ts)
        cp = np.zeros_like(ts)
        rebase = True 
        if self.skip: 
            step = self.rows
        else: 
            step = 1
        t = 0 
        while t <= len(ts)-step:
            if rebase: 
                end_base_matrix = t+self.rows*cols - 1 #last index in the base matrix 
                if end_base_matrix > len(ts)-step: #check
                    break
                base_matrix = np.reshape([ts[t:end_base_matrix+1]], (self.rows, cols), order="F")
                U,S,ـ = np.linalg.svd(base_matrix, full_matrices= False)
                if not self.rank: 
                    self.rank = utils.estimate_rank(base_matrix, 0.95)
                singular_values = S[:self.rank]
                perp_basis = U[:, self.rank:]

                singular_shift_c = self._estimate_singular_shift_c(base_matrix)
                distance_shift_c, eps = self._estimate_distance_shift_c(base_matrix)
                singular_h = self.singular_threshold * singular_shift_c
                distance_h = self.distance_threshold * eps
                t = end_base_matrix + step 
                rebase = False

            test_matrix = np.reshape([ts[t-self.rows*cols+1:t+1]], (self.rows, cols), order="F") 
            test_vector = test_matrix[:, -1]
            _,S, _ = np.linalg.svd(test_matrix, full_matrices= False)
            test_singular_values = S[:self.rank]
            #distance detection 
            D_t = (np.linalg.norm(perp_basis.T @ test_vector, 2))**2 - distance_shift_c
            distance_score[t:t+step] = D_t
            distance_cusum_score[t:t+step] = max(distance_cusum_score[t-1] + D_t, 0)
            #singular values detection 
            D_t = np.linalg.norm(test_singular_values - singular_values) - singular_shift_c
            singular_score[t:t+step] = D_t
            singular_cusum_score[t:t+step] = max(singular_cusum_score[t-1] + D_t, 0)
            
            if distance_cusum_score[t] >= distance_h:
                print("original detection")
                cp[t] = 1
                rebase=True 

            if singular_cusum_score[t] >= singular_h: 
                print("new detection")
                cp[t] = 1
                rebase=True 
            t = t+step
        self.cp = cp



