import os
import numpy as np
import pandas as pd
import algorithms.utils as utils
from sklearn.preprocessing import StandardScaler


class mssa: 

    def __init__(self, window_size, rows, distance_threshold, rank=0.95, 
                 training_ratio= 0.9, skip = False, normalize = False):
        self.rows = rows
        self.cols_ts = int(window_size/self.rows)
        self.window_size = self.cols_ts*self.rows
        self.rank = rank
        self.distance_threshold = distance_threshold
        self.training_ratio = training_ratio
        self.skip = skip
        self.normalize = normalize 

    def _check_param(self): 
        ## Window size must be greater than 1 
        if self.window_size<=1: 
            raise RuntimeError('Window size is not greater than 1')
        ## Number of rows must be less than or equal to half window size
        if self.rows > self.window_size/2: 
            raise RuntimeError('Number of rows is greater than half window size')
        ## threshold must be positive 
        if self.distance_threshold <= 0: 
            raise RuntimeError('Detection threshold must be positive')
        ## Rank should not be larger than base matrix dimentsions 
        if self.rank > 1:
            if self.rank > min(self.rows, self.cols): 
                raise RuntimeError('Rank is bigger then matrix size')
        ## Training ration must be less than 1
        if self.training_ratio >= 1: 
            raise RuntimeError('Training ratio must be less than 1')


    def _format_input(self, ts): 
        self.ts = np.array(ts) 
        self.no_ts = self.ts.shape[1]
        self.cols = self.cols_ts*self.no_ts
        self._check_param()

    def _estimate_c(self, matrix, r, folds = 10): 
        max_eps = 0
        for i in range(folds):
            cols_ts = int(self.cols_ts * (1-self.training_ratio))
            cols_ts = max(cols_ts, 1)
            cols = self.no_ts * cols_ts
            if matrix.shape[1] - cols < r: 
                raise RuntimeError("training matrix size is less than the rank")
            test_ind = np.random.choice(range(self.cols_ts), size=cols_ts, replace=False) * self.no_ts
            test_ind = [list(range(i,i+self.no_ts)) for i in test_ind]
            test_ind = np.sort(np.array(test_ind).flatten())
            base_matrix = np.delete(matrix, test_ind,1)
            test_matrix = matrix[:, test_ind]
            # Subspace estimation error (eps) 
            U,_,_ = np.linalg.svd(base_matrix, full_matrices=False)
            perp_basis = U[:,r:]
            proj = perp_basis.T @ test_matrix
            proj = np.linalg.norm(proj, 2, 0)
            proj = np.sum(proj.reshape(-1, self.no_ts), axis=1)
            eps = np.max(proj)**2
            if eps > max_eps: 
                max_eps = eps
        # Noise variance estimation
        U,S,VT = np.linalg.svd(matrix, full_matrices=False)
        noise_matrix = U[:, r:] @ np.diag(S[r:]) @ VT[r:, :]
        number_of_samples = noise_matrix.size
        variance = (1/(number_of_samples-1)) * np.linalg.norm(noise_matrix - np.mean(noise_matrix), 'fro')**2
        return (max_eps + self.no_ts*(self.rows-r)*variance), max_eps+1e-10

    def train(self, ts): 
        pass

    def detect(self, ts):

        self._format_input(ts)
        if self.skip: 
            step = self.rows
        else: 
            step = 1
        cp = np.zeros_like(self.ts[:, 0])
        if self.normalize:
            self.scaler = StandardScaler()
            self.ts = self.scaler.fit_transform(self.ts)
            
        t = 0
        rebase = True
        self.distance_score = np.zeros_like(self.ts[:,0])
        self.distance_cusum_score = np.zeros_like(self.ts[:,0])
        while t <= len(self.ts)-step:
            if rebase: 
                if t > len(self.ts) - step - self.window_size: 
                    break

                base_matrix = self.ts[t:t+self.window_size,:].reshape([self.rows, self.cols], order = 'F')
                base_matrix = base_matrix[:,np.arange(self.cols).reshape([self.no_ts,self.cols_ts]).flatten('F')]
                U,S,ـ = np.linalg.svd(base_matrix, full_matrices= False)
                if self.rank < 1: 
                    r = utils.estimate_rank(base_matrix, self.rank)
                else: 
                    r = self.rank
                singular_values = S[:r]
                perp_basis = U[:, r:]

                distance_shift_c, eps = self._estimate_c(base_matrix, r, folds = 500)
                distance_h = self.distance_threshold * eps
                t += self.window_size
                rebase = False
            
            test_matrix = self.ts[t-self.window_size:t].reshape([self.rows, self.cols], order = 'F')
            test_matrix = test_matrix[:,np.arange(self.cols).reshape([self.no_ts,self.cols_ts]).flatten('F')]
                
            test_vector = self.ts[t-self.rows:t, :]

            
            #distance detection 
            D_t = (np.linalg.norm(perp_basis.T @ test_vector, 2,0).sum())**2 - distance_shift_c
            self.distance_score[t:t+step] = D_t
            self.distance_cusum_score[t:t+step] = max(self.distance_cusum_score[t-1] + D_t, 0)

            if self.distance_cusum_score[t] >= distance_h:
                cp[t] = 1
                rebase=True 
                continue
            t = t+step
        self.cp = utils.convert_binary_to_intervals(cp, min_interval_length=2)


