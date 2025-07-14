import copy
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from EquationToMatrix import EqToMx
import time

class Tee(object):
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

class BlockingProbability:
    def __init__(self, core, mode, slot, classes, edgelist, lambda_val, mu_val, output, states, adj_m):
        self.core = core
        self.mode = mode
        self.slot = slot
        self.classes = classes
        self.lambda_val = lambda_val
        self.mu_val = mu_val
        self.states = states
        self.col = self.core * self.mode * 2
        self.row = self.slot
        self.edgeList = edgelist
        self.matrix = self.__init_resource_matrix()
        self.adj_mat = self.__create_graph()
        self.int_list = adj_m
    
    def __init_resource_matrix(self):
        return csr_matrix((self.col, self.slot), dtype=int)
    
    def __create_graph(self):
        return csr_matrix((self.col, self.col), dtype=int)
    
    def run(self, policy, count_out, count_inc, transition):
        policies = {'ff': self.firstFitExec, 'rf': self.randomFitExec}
        return policies.get(policy)(count_out, count_inc, transition)
    
    def update_matrix(self, row, col, value):
        mat_lil = self.matrix.tolil()
        mat_lil[row, col] = value
        self.matrix = mat_lil.tocsr()
    
    def firstFitExec(self, count_out, count_inc, transitionIncState):
        lambda_op, mu_op = 'L', 'mu'
        classesD = {self.classes[i]: f'{lambda_op}{i+1}' for i in range(len(self.classes))}
        mu = {self.classes[i]: f'{mu_op}{i+1}' for i in range(len(self.classes))}
        finalOutput = []
        for i in range(len(count_out)):
            eq = self.createEquation(classesD, mu, count_out[i], count_inc[i], transitionIncState, i+1, count_inc, finalOutput)
        l_ops = [f'{v}={self.lambda_val[i]}' for i, v in enumerate(classesD.values())]
        mu_ops = [f'{v}={self.mu_val[i]}' for i, v in enumerate(mu.values())]
        E = EqToMx(len(self.states), finalOutput, 'matrix.txt')
        outputMatrix, bp = E.run(l_ops, mu_ops, mu)
        return self.calcBP(outputMatrix, bp)
    
    def calcBP(self, output, bp):
        df = pd.DataFrame(output)
        dfy = np.zeros((len(self.states), 1))
        dfy[len(self.states) - 1] = 1
        res = np.linalg.solve(df, dfy)
        s = [sum(res[i] for i in range(len(bp)) if self.classes[j] in bp[i]) for j in range(len(self.classes))]
        N = sum(s[i] * self.lambda_val[i] for i in range(len(s)))
        D = sum(self.lambda_val)
        sigma = sum(((1 - s[i]) * self.lambda_val[i] * self.classes[i]) / self.mu_val[i] for i in range(len(s)))
        return N/D, sigma/(self.core * self.mode * self.slot * 2)

def calculateProb():
    classes = [2]
    edgeList = [(0, 1), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)]
    core, mode, slot, load_val = 3, 2, 4, 10
    adj_m = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    inps = [(i + 1) * 1 for i in range(load_val)]
    file_name = f'CPStates_{slot}_{classes}.csv'
    lambda_val = [(x + 1) * 0.1 for x in range(len(classes))]
    mu_val = [1 for _ in range(len(classes))]
    states = []
    mains = BlockingProbability(core, mode, slot, classes, edgeList, lambda_val, mu_val, 'ignore', states, adj_m)
    type_ = 'rf'
    count_out, count_inc = mains.randomFit() if type_ == 'ff' else mains.firstFit()
    finalD, transitionIncState = mains.transitionIncState(count_inc)
    for i in range(len(inps)):
        lambda_val = [(x + 1) * 0.1 * (i + 1) for x in range(len(classes))]
        mu_val = [1 for _ in range(len(classes))]
        mains = BlockingProbability(core, mode, slot, classes, edgeList, lambda_val, mu_val, 'ignore', states, adj_m)
        result = mains.run(type_, count_out, count_inc, transitionIncState)
        with open(file_name, 'a') as fF:
            fF.write(f"{i},{sum(lambda_val)/len(lambda_val)},{len(mains.states)},{core},{mode},{slot},{classes},{float(result[0])},{float(result[1])},{time.time()}\n")

calculateProb()
