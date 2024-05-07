from random import choice, random, sample
import numpy as np
import networkx as nx
from copy import deepcopy

class ArmBaseStruct(object):
    def __init__(self, armID):
        self.armID = armID
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward  = 0.0
        self.p_max = 1
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1
        self.averageReward = self.totalReward/float(self.numPlayed)

class UCB1Struct(ArmBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return float('inf')
        else:
            p = self.totalReward / float(self.numPlayed) + 1e-2*np.sqrt(3*np.log(allNumPlayed) / (2.0 * self.numPlayed))
            if p > self.p_max:
                p = self.p_max
                # print 'p_max'
            return p

             
class UCB1AlgorithmAttack:
    def __init__(self, G, P, parameter, seed_size, target_arms, oracle, feedback = 'edge', prop_dist=1):
        self.G = G
        self.trueP = P
        self.parameter = parameter  
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        #Initialize P
        self.currentP =nx.DiGraph()
        for (u,v) in self.G.edges():
            self.arms[(u,v)] = UCB1Struct((u,v))
            self.currentP.add_edge(u,v, weight=float('inf'))
        self.list_loss = []
        self.TotalPlayCounter = 0
        self.prop_dist = prop_dist

        self.target_arms = target_arms
        self.num_targetarm_played = []
        self.totalCost = []
        self.cost = []
        self.num_basearm_played = []

        target_prop_dict = {}
        self.target_prop_list = []
        for u in self.target_arms:
            target_prop_dict[u] = 1
            self.target_prop_list.append(u)
        for u in self.target_prop_list:
            if target_prop_dict[u] < self.prop_dist:
                diff = list(set(self.G.neighbors(u)).difference(set(self.target_prop_list)))
                for n in diff:
                    self.target_prop_list.append(n)
                    target_prop_dict[n] = target_prop_dict[u] + 1
        
    def decide(self):
        self.TotalPlayCounter +=1
        S = self.oracle(self.G, self.seed_size, self.currentP)        

        return S       

    def numTargetPlayed(self, S):
        num_basearm_played = 0
        num_targetarm_played = 0

        for u in S:
            if u in self.target_arms:
                num_basearm_played += 1

        if num_basearm_played == self.seed_size:
            num_targetarm_played = 1

        num_basearm_played = num_basearm_played*100/self.seed_size
        self.num_basearm_played.append(num_basearm_played)
        if len(self.num_targetarm_played) == 0:
            self.num_targetarm_played.append(num_targetarm_played)
        else:
            self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)

         
    def updateParameters(self, S, live_nodes, live_edges, iter_): 
        count = 0
        loss_p = 0 
        loss_out = 0
        loss_in = 0
        cost = 0
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    if u in self.target_prop_list:
                        self.arms[(u, v)].updateParameters(reward=1)
                    else:
                        self.arms[(u, v)].updateParameters(reward=0)
                        cost += 1
                else:
                    self.arms[(u, v)].updateParameters(reward=0)
                
                #update current P
                self.currentP[u][v]['weight'] = self.arms[(u,v)].getProb(self.TotalPlayCounter) 
                estimateP = self.currentP[u][v]['weight']
                trueP = self.trueP[u][v]['weight']
                loss_p += np.abs(estimateP-trueP)
                count += 1

        # print("-------------")
        self.list_loss.append([loss_p/count])
        if len(self.totalCost) == 0:
            self.totalCost = [cost]
        else:
            self.totalCost.append(self.totalCost[-1] + cost)
        self.cost.append(cost)
        self.numTargetPlayed(S)

    def getLoss(self):
        return np.asarray(self.list_loss)

    def getP(self):
        return self.currentP