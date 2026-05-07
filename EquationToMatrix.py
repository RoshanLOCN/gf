import numpy as np


class EqToMx:
    def __init__(self, size, finalOutput, matrixPath):
        self.size = size
        self.finalOutput = finalOutput
        self.matrix = np.zeros((size, size), dtype=np.float64)
        self.MatrixPath = matrixPath

    def run(self, l, m, mu_op):
        bp = self.createMatrix(l, m, mu_op)
        return self.matrix, bp

    def createMatrix(self, l, m, mu_op):
        ns = {}
        for i in l:
            exec(i, ns)
        for i in m:
            exec(i, ns)

        outputEq = self.finalOutput
        bpList = []
        rhs_vector = [0] * self.size

        classes = {v: k for k, v in mu_op.items()}

        for i in range(len(outputEq)):
            eq = outputEq[i]
            outR = eq.get('outR', [])
            if outR:
                self.setVal(i, i, eval('+'.join(outR), ns))

            for inc in eq.get('incR', []):
                for k, v in inc.items():
                    for idx in v:
                        self.setVal(i, idx, eval('-' + k, ns))

            for elist in eq.get('EincR', []):
                for k, v in elist.items():
                    self.setVal(i, v[1] - 1, eval('-' + k + '/' + v[0], ns))

            if 'rhs_const' in eq:
                rhs_vector[i] = -eq['rhs_const']

            try:
                if eq.get('incR'):
                    used_keys = {k for item in eq['incR'] for k in item}
                    bpList.append([classes[k] for k in set(classes) - used_keys])
                else:
                    bpList.append(list(classes.values()))
            except Exception:
                bpList.append([])

            if i == self.size - 1:
                self.matrix[i, :] = 1.0
                rhs_vector[i] = 1

        return bpList, rhs_vector

    def setVal(self, row, col, val):
        self.matrix[row, col] = val

    def saveMatrix(self):
        pass
