import numpy as np
import random as rd
import copy as cp

class Simplex:
    def __init__(self, A : np.array, b : np.array, c : np.array):
        '''
        Solving a simple lp using simplex method(assuming the model has only one solution if the solution exists)
            max{cx}
            AX <= b
        relative statistical indicator:
            1. x_B : B(-1) * b
            2. x_N : 0
            3. reduce costs(for nonbass) : c_N - C_B * B(-1) * N
            4. obj : C_B * B(-1) * b
        Parameters
            A : coefficient matrix
            b : constraint bound vector
            c : value vector
        '''
        self.A = A
        self.b = b
        self.c = c
        self._A = None
        self._b = None
        self._c = None
    
        self.rows_num = self.A.shape[0] # rows count
        self.cols_num = self.A.shape[1] # columns count
        self._cols_num = -1

        self._B = None
        self._B_inverse = None
        self._N = None
        self._c_B = None
        self._c_N = None
        self.base_indexs = None # set
        self.nonbase_indexs = None # set
        self.reduce_costs = None

        self._x_B = None
        self.sol = Solution()

        self.standardize()
    
    def solve(self, infos_show=True):
        '''
        solve the model using simplex algorithm
        '''
        # initialize the bases
        self.gene_initial_bases_indexs()
        self.update()        
        # print('reduce costs:%s' % self.reduce_costs)

        while(sum(self.reduce_costs > 0) != 0):
            # select a variable to be inserted into the bases(strategy : random)
            insert_index = list(self.nonbase_indexs)[rd.choice(np.where(self.reduce_costs > 0)[0])]
            
            # select a variable to be removed from the bases
            u = np.dot(self._B_inverse, self._A[:,insert_index])
            if(sum(u > 0) == 0):
                # which means the optimum is infinity
                self.sol.state = 'infinity'
                self.sol.objective = float('inf')
                break
            remove_index = -1
            sitas = list()
            for i in range(u.shape[0]):
                if(u[i] <= 0):
                    sitas.append(float('inf'))
                else:
                    sitas.append(self._x_B[i] / u[i])
            remove_index = list(self.base_indexs)[np.argmin(np.array(sitas))]

            # update the bases and nonbases
            self.base_indexs.remove(remove_index)
            self.base_indexs.add(insert_index)
            self.nonbase_indexs.remove(insert_index)
            self.nonbase_indexs.add(remove_index)
            self.update()
            
            if(infos_show):
                print('-----------------------------------------------------------')
                print('insert index:%d' % insert_index)
                print('sitas:%s' % sitas)
                print('remove index:%d' % remove_index)
                print('reduce costs:%s' % self.reduce_costs)

        # generate the final solution of problem
        self.sol.state = 'global optimum'
        self.sol.objective = self._c_B.dot(self._B_inverse).dot(self._b)
        self.sol.variables_value = np.zeros(self.cols_num)
        for i, variable_index in enumerate(self.base_indexs):
            if(variable_index < self.cols_num):
                self.sol.variables_value[variable_index] = self._x_B[i]

    def update(self):
        '''
        update the infos(matrix and vector)
        '''
        list_base_indexs = list(self.base_indexs)
        list_nonbase_indexs = list(self.nonbase_indexs)
        self._c_B = self._c[list_base_indexs]
        self._c_N = self._c[list_nonbase_indexs]
        self._B = self._A[:,list_base_indexs]
        self._N = self._A[:,list_nonbase_indexs]
        self._B_inverse = np.linalg.inv(self._B)
        
        self._x_B = np.dot(self._B_inverse, self._b)
        self.reduce_costs = self._c_N - self._c_B.dot(self._B_inverse).dot(self._N)

    def gene_initial_bases_indexs(self):
        '''
        set the relaxation variables to the initial basic variables directly(because the porblem is
        a maximul problem and the constraints' sense are '<=')
        '''
        self.base_indexs = set([i for i in range(self.cols_num, self._cols_num)])
        self.nonbase_indexs = set(range(self._cols_num)) - self.base_indexs
        if(len(self.base_indexs) != self.rows_num):
            raise Exception('the number of initial bases should be equal to the constraints number')
        # print(list(self.base_indexs))
        # print(list(self.nonbase_indexs))

    def standardize(self):
        '''
        standardize the model
        '''
        self._b = self.b
        self._c = np.append(self.c, [0] * self.rows_num)
        A_list = self.A.tolist()
        for index in range(self.rows_num):
            A_list[index] += [0] * index
            A_list[index] += [1]
            A_list[index] += [0] * (self.rows_num - 1 - index)
        self._A = np.array(A_list)
        
        self._cols_num = self._A.shape[1]

class Solution:
    def __init__(self):
        self.objective = None
        self.state = None # global optimum, infinity, non-feasible
        self.variables_value = None
    def show(self):
        if(self.state == 'global optimum'):
            print('global optimum has been found, that is:%f' % self.objective)
            print('variables value:%s' % self.variables_value)
        if(self.state == 'infinity'):
            print('the problem has infinite solution')
        if(self.state == 'non-feasible'):
            print('the problem is non-feasible')

if(__name__ == '__main__'):
    A = np.array([1,2,2,1]).reshape(2,2)
    b = np.array([3,3])
    c = np.array([1,1])
    simplex = Simplex(A, b, c)
    simplex.solve()
    simplex.sol.show()

    print('finished')