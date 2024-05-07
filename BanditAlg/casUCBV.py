from math import *
import numpy as np
import random


class CascadeUCB_V():

    def __init__(self, dataset, num_arms, seed_size, target_arms):
        self.dataset = dataset
        self.T = {}
        self.w_hat = {}
        self.U = {}
        self.t = 1
        self.num_arms = num_arms
        self.seed_size = seed_size
        self.target_arms = target_arms

        for i in range(self.num_arms):
            self.T[i] = 0
            self.w_hat[i] = 0

    def decide(self):
        for i in range(self.num_arms):
            v = self.w_hat[i] * (1-self.w_hat[i])
            if self.T[i] == 0:
                self.U[i] = float('inf')
            else:
                self.U[i] = self.w_hat[i] + (np.sqrt(4*v*np.log(self.t)/self.T[i]) + 6*np.log(self.t)/self.T[i])
                self.U[i] = min(1, max(self.U[i], 0))
        
        best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]
        
        self.click_prob = 1
        for i in best_arms:
            self.click_prob *= (1 - self.dataset.w[i])

        return best_arms

    
    def updateParameters(self, C, best_arms):
        if C == -1:
            C = self.num_arms

        for i in range(self.seed_size):
            if i <= C:
                arm = best_arms[i]
                
                r = self.T[arm]*self.w_hat[arm]
                if i == C:
                    r += 1
                self.w_hat[arm] = r/(self.T[arm]+1)

                self.T[arm] += 1
            
            else:
                break

        self.t += 1