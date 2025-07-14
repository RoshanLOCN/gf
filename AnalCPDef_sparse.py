import copy
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from EquationToMatrix_Sparse import EqToMx
import time
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure it's written immediately

    def flush(self):
        for f in self.files:
            f.flush()

class BlockingProbability:
    def __init__(self,
                 core,
                 mode,
                 slot,
                 classes,
                 edgelist,
                 lambda_val,
                 mu_val,
                 output,
                 states,
                 adj_m
                 ):
      self.core = core
      self.mode = mode
      self.slot = slot
      self.classes = classes
      self.lambda_val = lambda_val
      self.mu_val = mu_val
      self.states = states
      self.col = self.core*self.mode*2
      self.row = self.slot
      self.edgeList = edgelist
      self.matrix = self.__init_resource_matrix()
      self.adj_mat = self.__create_graph()
      self.int_list=adj_m
      self.__output_file=f'{output}_classes_{len(self.classes)}_core_{self.core}_mode_{self.mode}_slot_{self.slot}.txt'
      self.__matJson = f'matrixJson_{output}_classes_{len(self.classes)}_core_{self.core}_mode_{self.mode}_slot_{self.slot}.json'
      self.__matrix = f'matrix_{output}_classes_{len(self.classes)}_core_{self.core}_mode_{self.mode}_slot_{self.slot}.txt'

    def run(self, policy, count_out, count_inc, transition):
        policies = {
            'ff': self.firstFitExec,
            'rf': self.randomFitExec
        }
        return policies.get(policy)(count_out, count_inc, transition)

    def randomFitExec(self, count_out, count_inc, transitionIncState):
        # count_out, count_inc = self.randomFit()
        # finalD, transitionIncState = self.transitionIncState(count_inc)
        # # print(len(self.states))
        # for i in range(len(self.states)):
        #     print(i)
        #     self.mprint(self.states[i])
        #     print()
        lambda_op = 'L'
        mu_op = 'mu'
        classesD = {}
        mu = {}
        for i in range(len(self.classes)):
            classesD[self.classes[i]] = lambda_op+str(i+1)
        for i in range(len(self.classes)):
            mu[self.classes[i]] = mu_op+str(i+1)
        finalOutput = []
        # f = open(self.__output_file,'w')
        for i in range(len(count_out)):
            eq = self.createEquation(classesD, mu, count_out[i], count_inc[i], transitionIncState, i+1, count_inc, finalOutput)
            # f.write(str(i)+':'+eq+'\n')
        # f = open(self.__matJson, 'w')
        fSt = str(finalOutput).replace("'",'"')
        # f.write(fSt)
        l_ops = []
        j = 0
        for i in list(classesD.values()):
            l_ops.append(str(i)+'='+str(self.lambda_val[j]))
            j+=1
        mu_ops = []
        j = 0
        for i in list(mu.values()):
            mu_ops.append(str(i)+'='+str(self.mu_val[j]))
            j+=1
        E = EqToMx(len(self.states), finalOutput, self.__matrix)
        outputMatrix, bp = E.run(l_ops, mu_ops, mu)
        print("Here is BP-For RF", len(bp)- bp.count([]))
        r = self.calcBP(outputMatrix, bp)
        return r

    def firstFitExec(self, count_out, count_inc, transitionIncState):

        lambda_op = 'L'
        mu_op = 'mu'
        classesD = {}
        mu = {}
        for i in range(len(self.classes)):
            classesD[self.classes[i]] = lambda_op+str(i+1)
        for i in range(len(self.classes)):
            mu[self.classes[i]] = mu_op+str(i+1)
        finalOutput = []
        # f = open(self.__output_file,'w')
        for i in range(len(count_out)):
            eq = self.createEquation(classesD, mu, count_out[i], count_inc[i], transitionIncState, i+1, count_inc, finalOutput)
            # f.write(str(i)+':'+eq+'\n')
        # f = open(self.__matJson, 'w')
        fSt = str(finalOutput).replace("'",'"')
        # f.write(fSt)
        l_ops = []
        j = 0
        for i in list(classesD.values()):
            l_ops.append(str(i)+'='+str(self.lambda_val[j]))
            j+=1
        mu_ops = []
        j = 0
        for i in list(mu.values()):
            mu_ops.append(str(i)+'='+str(self.mu_val[j]))
            j+=1
        E = EqToMx(len(self.states), finalOutput, self.__matrix)
        outputMatrix, bp = E.run(l_ops, mu_ops, mu)
        r = self.calcBP(outputMatrix, bp)
        print("Here is BP-For FF", len(bp)- bp.count([]))
        return r

    def __init_resource_matrix(self):
        return lil_matrix((self.col, self.slot), dtype=int)

    def __create_graph(self):
        return [[0 for i in range(self.col)] for j in range(self.col)]

    def __str__(self) -> str:
        return "Blocking Probability based on states"

    def create_edges(self):
        for edge in self.edgeList:
            row,column=edge[0],edge[1]
            self.adj_mat[row][column]=1

    def mprint(self, matrix):
        for row in matrix:
            print(*row)

    def RandomFitfillAdj(self, row, col, size, flip):
      a=[row, col, size, flip]
      if col == self.col:
        return
      coreCol = self.isLastCoreCol(col)
      try:
        if coreCol:
            for j in range(1, self.core):
                if col+(j*self.mode) >= self.col:
                    break
                if self.isAdj(col, col+(j*self.mode)):
                    # print(col, j+self.mode)5
                    flag = False
                    for i in range(size):
                        if flip:
                            if self.matrix[row-i][col+(j*self.mode)] != 1:
                                self.matrix[row-i][col+(j*self.mode)] = -1
                        else:
                            if self.matrix[row+i][col+(j*self.mode)] != 1:
                                self.matrix[row+i][col+(j*self.mode)] = -1
        else:
            if self.isAdj(col, col+1):
                flag = False
                for i in range(size):
                    if flip:
                        if self.matrix[row-i][col+1]!=1:
                            self.matrix[row-i][col+1] = -1
                    else:
                        if self.matrix[row+i][col+1]!=1:
                            self.matrix[row+i][col+1] = -1

            for j in range(1, self.core):
                if col+(j*self.mode) >= self.col:
                    break
                if self.isAdj(col, col+(j*self.mode)):
                    flag = False
                    for i in range(size):
                        if flip:
                            if self.matrix[row-i][col+(j*self.mode)] != 1:
                                self.matrix[row-i][col+(j*self.mode)] = -1
                        else:
                            if self.matrix[row+i][col+(j*self.mode)] != 1:
                                self.matrix[row+i][col+(j*self.mode)] = -1
      except:
          pass

    def interferncefillAdj(self, row, col, size, flip):
        a=[row, col, size, flip]
        # # print(row, col)
        # # coreCol = self.isLastCoreCol(col)
        try:
                try:
                    if self.int_list[row] % 2 != 0 and self.int_list[row] != 0:
                        for i in range(size):
                            self.matrix[row + 1, col:col + size] =  -1  # Copy row down
                    elif self.int_list[row] % 2 == 0 and self.int_list[row] != 0:
                        for i in range(size):
                            self.matrix[row - 1, col:col + size] = -1  # Copy row down
                except Exception as e:
                    pass

        except:
            pass

    def isLastCoreCol(self, col):
      cores = [i*self.mode for i in range(self.core+1)]
      cores = cores[1:]
      for i in cores:
        if col < i-1:
          return False
        if col == i-1:
          return True

    def isAdj(self, col1, col2):
      if self.adj_mat[col1][col2] == 1:
        return True
      return False

    def createRandomFitInitialStates(self):
        currInd = 0
        self.states.append(self.matrix)
        while currInd <= len(self.states)-1:
            for j in range(len(self.states[currInd])):
                for x in self.classes:
                    valueChecked = [0]*x
                    for state_check in range(len(self.states[currInd][j])):
                        # print(state[i][j][state_check : state_check+x], valueChecked, state_check, x)
                        if self.states[currInd][j][state_check :  state_check+x] == valueChecked:
                            tempMatrix = copy.deepcopy(self.states[currInd])
                            tempMatrix[j][state_check :  state_check+x] = [1]*x
                            self.matrix = tempMatrix
                            self.matrix = np.transpose(self.matrix)
                            self.RandomFitfillAdj(state_check, j, x , False)
                            self.matrix = np.flip(self.matrix)
                            self.RandomFitfillAdj(self.row-state_check-1, self.col-j-1, x,True)
                            self.matrix = np.flip(self.matrix)
                            self.matrix = np.transpose(self.matrix)
                            tempMatrix = self.matrix.tolist()
                            if tempMatrix in self.states:
                                continue
                            self.states.append(tempMatrix)
            currInd+=1

    def randomFit(self):
        self.createRandomFitInitialStates()
        # print(self.states)
        blocked_states = []
        cd=[]
        n=2
        for matrix in self.states:
            is_valid = False
            for row in matrix:
                if '0' * n  in ''.join(map(str, row)):
                    is_valid = True
                    break
            if not is_valid:
                blocked_states.append(matrix)
        # print(blocked_states)
        a = self.defragment_zeros(blocked_states)
        for i in range(len(a)):
            v1=[]
            for j in range(len(a[i])):
                    v = [[0 for _ in range(self.col)] for _ in range(self.row)]
                    res = [[0 if num == -1 else num for num in row] for row in a[j]]
                    transposed_matrix = [[row[i] for row in res] for i in range(len(res[0]))]
                    for k in range(self.col):
                        for l in range(self.row):
                            c = 0
                            if transposed_matrix[l][k]==1:
                                v[l][k] = 1
                                while c<self.col:
                                    if c!=k and self.isAdj(k,c):
                                        v[l][c]=-1
                                    c=c+1
            v[:] = [[row[i] for row in v] for i in range(len(v[0]))]
            (self.states).append([row[:] for row in v])
            # k.append([row[:] for row in v1])
            # print(1)
        self.createRandomFitInitialStates(self.states)
        slotArr = []
        transitionArr = []
        for currInd in range(len(self.states)):
            slotValue = {}
            finalClassesTransition = {}
            for j in range(len(self.states[currInd])):
                for x in self.classes:
                    valueChecked = [0]*x
                    for state_check in range(len(self.states[currInd][j])):
                        # print(state[i][j][state_check : state_check+x], valueChecked, state_check, x)
                        if self.states[currInd][j][state_check :  state_check+x] == valueChecked:
                            if x in slotValue.keys():
                                slotValue[x] += 1
                            else:
                                slotValue[x] = 1
                            tempMatrix = copy.deepcopy(self.states[currInd])
                            tempMatrix[j][state_check :  state_check+x] = [1]*x
                            self.matrix = tempMatrix
                            self.matrix = np.transpose(self.matrix)
                            self.RandomFitfillAdj(state_check, j, x , False)
                            self.matrix = np.flip(self.matrix)
                            self.RandomFitfillAdj(self.row-state_check-1, self.col-j-1, x,True)
                            self.matrix = np.flip(self.matrix)
                            self.matrix = np.transpose(self.matrix)
                            tempMatrix = self.matrix.tolist()
                            if tempMatrix in self.states:
                                ind = self.states.index(tempMatrix)
                                if x in finalClassesTransition.keys():
                                    finalClassesTransition[x].append(ind)
                                else:
                                    finalClassesTransition[x] = [ind]
            slotArr.append(slotValue)
            transitionArr.append(finalClassesTransition)
        return slotArr, transitionArr

    def transitionIncState(self, count_inc):
        finalD = {}
        transitionIncomingState = dict()
        for j in range(len(count_inc)):
            for k, v in count_inc[j].items():
                for i in v:
                    if i in finalD.keys():
                        finalD[i] += 1
                    else:
                        finalD[i] = 1

                    if i in transitionIncomingState.keys():
                        if k in transitionIncomingState[i].keys():
                            transitionIncomingState[i][k].append(j)
                        else:
                            transitionIncomingState[i][k] = [j]
                    else:
                        transitionIncomingState[i] = dict()
                        transitionIncomingState[i][k] = [j]
        return finalD, transitionIncomingState

    def createFirstFitInitialStates(self):
        currInd = 0
        self.states.append(self.matrix.copy())  # Ensure copying correctly

        while currInd <= len(self.states) - 1:
            for j in range(self.states[currInd].shape[0]):  # FIX: Use .shape[0] instead of len()
                for x in self.classes:
                    valueChecked = 0  # Since nnz returns a number, compare it with an integer
                    state_check = 0
                    while state_check <= self.states[currInd].shape[1] - x:  # FIX: Use .shape[1] for columns
                        if self.states[currInd][j, state_check: state_check + x].getnnz() == valueChecked and \
                                state_check + x < self.slot and \
                                self.states[currInd][j, state_check + x] == 0:

                            tempMatrix = self.states[currInd].copy()
                            tempMatrix[j, state_check: state_check + x] = 1  # Assign values properly

                            self.matrix = tempMatrix
                            self.interferncefillAdj(j, state_check, x, False)

                            state_check += x
                            if tuple(tempMatrix.tocoo().data) in [tuple(s.tocoo().data) for s in self.states]:
                                continue
                            self.states.append(tempMatrix)
                        else:
                            state_check += 1

                        # Second condition, keeping structure same
                        if self.states[currInd][j, state_check: state_check + x].getnnz() == valueChecked and \
                                state_check + x == self.slot:

                            tempMatrix = self.states[currInd].copy()
                            tempMatrix[j, state_check: state_check + x] = 1  # Assign values properly

                            self.matrix = tempMatrix
                            self.interferncefillAdj(j, state_check, x, False)

                            state_check += x
                            if tuple(tempMatrix.tocoo().data) in [tuple(s.tocoo().data) for s in self.states]:
                                continue
                            self.states.append(tempMatrix)
                        else:
                            state_check += 1
            currInd += 1

    def firstFit(self):
        self.createFirstFitInitialStates()
        slotArr = []
        transitionArr = []
        for currInd in range(len(self.states)):
            slotValue = {}
            finalClassesTransition = {}
            for j in range(self.row):
                for x in self.classes:
                    for state_check in range(self.slot - x + 1):
                        if self.states[currInd][j, state_check:state_check + x].nnz == 0:
                            slotValue[x] = slotValue.get(x, 0) + 1
                            tempMatrix = self.states[currInd].copy()
                            tempMatrix[j, state_check:state_check + x] = 1
                            # if state_check + x < self.slot: tempMatrix[j, state_check + x] = -1
                            # tempMatrix[j, state_check + x + 1] = 1
                            self.matrix = tempMatrix
                            self.interferncefillAdj(j, state_check, x, False)
                            tempMatrix = self.matrix.copy()
                            if any((tempMatrix != state).nnz == 0 for state in self.states):
                                ind = next((i for i, state in enumerate(self.states) if (tempMatrix != state).nnz == 0),-1)
                                finalClassesTransition.setdefault(x, []).append(ind)
            slotArr.append(slotValue)
            transitionArr.append(finalClassesTransition)
        return slotArr, transitionArr

    def createEquation(self, classes, mu, count, transition, globalTransition, ind, count_inc_dict, finalOutput):
        equation = '('
        equation+='('
        tempD = {}
        l = []
        for i in count.keys():
            l.append(classes.get(i))

        le = '+'.join(l)
        equation+=le
        tl = []
        if globalTransition.get(ind-1):
            for k, v in globalTransition.get(ind-1).items():
                tl.append(str(len(v))+'*'+mu.get(k))

        tempD['outR'] = l+tl
        tlst = '+'.join(tl)
        if tl != []:
            if l!= []:
                equation+="+("
                equation+=tlst
                equation+=")"
            else:
                equation+=tlst
        equation+=')'
        equation+= ' * P'+str(ind)+')'
        if l != []:
            equation+= '-'
        size = len(transition.keys())
        for k, v in transition.items():
            equation+=mu.get(k)+'('
            equation+='+'.join(map(str, v))
            equation+=')'
            if 'incR' in tempD.keys():
                tempD['incR'].append({mu.get(k):v})
            else:
                tempD['incR'] = [{mu.get(k):v}]
            size-=1
            if size>0:
                equation+='-'

        if globalTransition.get(ind-1):
            for k, v in globalTransition.get(ind-1).items():
                for j in v:
                    equation += "- (("+classes.get(k) +'/'+ str(len(count_inc_dict[j][k])) +") * "+ 'P'+str(j+1) +")"
                    if 'EincR' in tempD.keys():
                        tempD['EincR'].append({classes.get(k):[str(len(count_inc_dict[j][k])), j+1]})
                    else:
                        tempD['EincR'] = [{classes.get(k):[str(len(count_inc_dict[j][k])), j+1]}]

        equation+='=0'

        finalOutput.append(tempD)
        return equation

    def calcBP(self, output, bp):
            df = pd.DataFrame(output)
            dfy = np.zeros((len(self.states), 1))
            dfy[-1] = 1
            epsilon = 1e-8

            if np.linalg.cond(df) > 1e15:  # Check condition number
                print("Warning: Ill-conditioned matrix detected, using pseudoinverse.")
                df += np.eye(df.shape[0]) * epsilon  # Add small value to diagonal
                res = np.linalg.pinv(df) @ dfy  # Use pseudo-inverse
            else:
                try:
                    res = np.linalg.solve(df, dfy)  # Solve system normally
                except np.linalg.LinAlgError:
                    print("Warning: Singular matrix detected, using pseudoinverse.")
                    df += np.eye(df.shape[0]) * epsilon
                    res = np.linalg.pinv(df) @ dfy

            s = np.zeros(len(self.classes))
            for i in range(len(bp)):
                for j in range(len(self.classes)):
                    if self.classes[j] in bp[i]:
                        s[j] += res[i, 0]  # Extract single element explicitly

            print("Summed Probabilities:", s)

            # # Handle singular matrix issue
            # try:
            #     res = np.linalg.solve(df, dfy)
            # except np.linalg.LinAlgError:
            #     print("Warning: Singular matrix detected, using pseudoinverse.")
            #     res = np.linalg.pinv(df) @ dfy  # Use pseudoinverse if singular
            #
            # s = np.zeros(len(self.classes))
            #
            # for i in range(len(bp)):
            #     for j in range(len(self.classes)):
            #         if self.classes[j] in bp[i]:
            #             s[j] += res[i]
            #
            # print("Summed Probabilities:", s)

            N = sum(s[i] * self.lambda_val[i] for i in range(len(s)))
            D = sum(self.lambda_val)

            if D == 0:  # Prevent division by zero
                print("Warning: Lambda sum (D) is zero, returning zero probability")
                return 0, 0

            N2 = sum(((1 - s[i]) * self.lambda_val[i] * self.classes[i]) / self.mu_val[i] for i in range(len(s)))

            b_p = N / D
            sigma = N2
            r = sigma / (self.core * self.mode * self.slot * 2)

            print("BLOCKING PROBABILITY:", b_p)
            print("RESOURCE UTILIZATION:", r)

            return b_p, r

def calculateProb():
    f = open('CP_defoutput_2_1.txt', 'w')
    # original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = Tee(sys.stdout, f)
    classes = [2]
    edgeList = [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2),(4, 5), (4, 6), (5, 4), (5, 7), (6, 4), (6, 7), (7, 5), (7, 6)]
    # [(0,1),(0,2),(0,4),(1,0),(1,3),(1,5),(2,0),(2,3),(2,4),(3,1),(3,2),(3,5),(4,0),(4,2),(4,5),(5,1),(5,3),(5,4)]
    core = 3
    mode = 2
    slot = 4
    load_val = 10
    adj_m=[0,1,2,0,1,2,0,1,2,0,1,2]
    inps = [(i+1)*1 for i in range(load_val)]
    # original_stdout = sys.stdout
    # f = open('output.txt', 'w')
    # sys.stdout = f
    file_name = 'CPStates_'+str(slot)+'_'+str(classes)
    fF = open(file_name+'.csv', 'a')
    fF.write('ind, load, states, core, mode, slots, classes, blocking probability, resource utilization, time taken\n')
    fF.close()

    lambda_val = [(x+1)*float(0.1) for x in range(len(classes))]
    mu_val = [1 for i in range(len(classes))]
    ro_ = [lambda_val[i]/mu_val[i] for i in range(len(classes))]
    load = sum(ro_)/len(ro_)
    fileName = 'ignore'
    start = time.time()
    states = []
    mains=BlockingProbability(core,
                            mode,
                            slot,
                            classes,
                            edgeList,
                            lambda_val,
                            mu_val,
                            fileName,
                            states,
                              adj_m
                              )
    type_ = 'ff'
    if type_ == 'ff':
        count_out, count_inc = mains.firstFit()
        finalD, transitionIncState = mains.transitionIncState(count_inc)
    else:
        count_out, count_inc = mains.randomFit()
        finalD, transitionIncState = mains.transitionIncState(count_inc)

    states = copy.deepcopy(mains.states)

    for i in range(len(inps)):
        fF = open(file_name+'.csv', 'a')
        lambda_val = [(x+1)*float(0.1)*(i+1) for x in range(len(classes))]
        mu_val = [1 for i in range(len(classes))]
        ro_ = [lambda_val[i]/mu_val[i] for i in range(len(classes))]
        load = sum(ro_)/len(ro_)
        fileName = 'ignore'
        mains=BlockingProbability(core,
                                mode,
                                slot,
                                classes,
                                edgeList,
                                lambda_val,
                                mu_val,
                                fileName,
                                states,
                              adj_m
                                  )
        result = mains.run(type_, count_out, count_inc, transitionIncState)
        end = time.time()
        res_st = str(i)+\
                ','+\
                str(load)+\
                ','+\
                str(len(mains.states))+\
                ','+\
                str(core)+\
                ','+\
                str(mode)+\
                ','+\
                str(slot)+\
                ','+\
                str(classes)+\
                ','+\
                str(float(result[0]))+\
                ','+\
                str(float(result[1]))+\
                ','+\
                str(end-start)+\
                '\n'

        fF.write(res_st)
        fF.close()

calculateProb()
# print("ggj")

# [[ 0  1  1 -1], [ 0  0  1  1], [ 1  1 -1  0], [ 0  1  1 -1], [ 0  0  0  0], [ 0  0  0  0], [ 0  0  0  0]]
# [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]