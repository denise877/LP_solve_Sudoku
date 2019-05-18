import os
print(os.listdir("../content"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization
import cvxpy as cp
from cvxpy import norm, pnorm




def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] =1
    rowR = np.zeros(N)
    rowR[0] =1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))
    
    colR = np.kron(np.ones((1,N)), rowC)
    col  = scl.toeplitz(rowC, colR)
    COL  = np.kron(col, np.eye(N))
    
    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0]=1
    boxR = np.kron(np.ones((1, M)), boxC) 
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))
    
    cell = np.eye(N**2)
    CELL = np.kron(cell, np.ones((1,N)))
    
    return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))

# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(input_quiz, N=9):
    m = np.reshape([int(c) for c in input_quiz], (N,N))
    r, c = np.where(m.T)
    v = np.array([m[c[d],r[d]] for d in range(len(r))])
    
    table = N * c + r
    table = np.block([[table],[v-1]])
    
    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N**3))
    for i in range(len(table.T)):
        CLUE[i,table[0,i]*N + table[1,i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr() 
    
    return CLUE
  
  
  
def cross_entropy_minimization(A):
    '''Using minimization of cross entropy function to find optimal solution of sudoku'''
    x = cp.Variable(A.shape[1])
    #print(x)
    prob = cp.Problem(cp.Minimize(0),
                     [A@x == 1, x>=0])
    
    prob.solve()
    x = x.value
    #print(x)
    #print(type(x))
    x = set_lower_bound(x)
    x_round = np.zeros(A.shape[1])
    #print(x_round)
    for i in range(len(x)):
        if x[i] >= 0.5 and x[i] <= 1:
            x_round[i] = 1
        else:
            x_round[i] = 0
            
            
    #print(x_round)
            
    #print(np.dot(A, x_round).all() != 1)
            
    alpha = 10
    
    #print(np.dot(A, x_round))
    #print(2222222)

    p=0        
    while np.dot(A, x_round).all() != 1:
        p+=1
        if p >=100:
            break
        #print(np.dot(A, x_round))
        #print(1111111)
        grad = gradient_cross_entropy(x)
        
        #print(grad)
        z = cp.Variable(A.shape[1])
        #print(np.dot(grad, z.value))
        obj = pnorm(alpha * grad - z, 2)
        
        prob = cp.Problem(cp.Minimize(obj),
                     [A@(x+z) == 1, (x+z)>=0])            
        prob.solve()
        #print(norm(alpha * grad - z, 1))
        x_record = x_round
        
        z = z.value
        #print(z)
        #print(norm(z,1))
        
        if np.linalg.norm(z) != 0:
            #print(111111)

#            prob = cp.Problem(cp.Minimize(np.dot(grad, z)),
#                        [A@(x+z) == 1, (x+z)>=0])
#            prob.solve()
            x = x + z
            x = set_lower_bound(x)
            #print(x)
            x_round = round_x(x)
            #print(x_record == x_round)
            
            
        else:
            #print(2222222)
            obj = pnorm(alpha * grad - z, 2)
            new_prob = cp.Problem(cp.Minimize(-1*obj),
                         [A@(x+z) == 1, (x+z)>=0])

            new_prob.solve()
            z = z.value
            #new_x = x + z
            x = x + z
            x = set_lower_bound(x)
           
            
            x_round = round_x(x)
            #print(x_record == x_round)
    #print(333333)
           
            
            
            
    return x_round
   
        
        
def round_x(x):
    '''Round x to 1 if 0.5<= x <=1.0, o otherwise'''
    x_round = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i] >= 0.5 and x[i] <= 1:
            x_round[i] = 1
        else:
            x_round[i] = 0
            
    return x_round    
      
def cross_entropy(x):
    '''Calculate the cross entropy of vector x'''
    s = -1 * np.dot(x, np.log(x))
    #print(s)
    
    return s
  
def gradient_cross_entropy(x):
    '''calculate gradient of cross entropy'''
    grad = -1 * np.log(x) -1
    
    return grad
  
        
def set_lower_bound(x):
    '''Given the x is a vector, if some entry of x is less than 1e-8, then set this 
    entry to 1e-8'''
    for i in range(len(x)):
        if x[i] < 1e-8:
            x[i] = 1e-8
            
            
    return x
          
  
import time

# We test the following algoritm on small data set.
data = pd.read_csv("../content/small2.csv") 

corr_cnt = 0
start = time.time()

random_seed = 42
np.random.seed(random_seed)

if len(data) > 1000:
    samples = np.random.choice(len(data), 1000)
else:
    samples = range(len(data))

for i in range(len(samples)):
    quiz = data["quizzes"][samples[i]]
    solu = data["solutions"][samples[i]]
    A0 = fixed_constraints()
    A1 = clue_constraint(quiz)

    # Formulate the matrix A and vector B (B is all ones).
    A = scs.vstack((A0,A1))
    A = A.toarray()
    B = np.ones(A.shape[0])
    '''
    
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    K = np.sum(s > 1e-12)
    S = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0]-K))])
    A = S@vh
    B = u.T@B
    B = B[:K]

    c = np.block([ np.ones(A.shape[1]), np.ones(A.shape[1]) ])
    G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],\
                         [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
    h = np.zeros(A.shape[1]*2)
    H = np.block([A, -A])
    b = B

    ret = sco.linprog(c, G, h, H, b, method='interior-point', options={'tol':1e-6})
    x = ret.x[:A.shape[1]] - ret.x[A.shape[1]:]
    '''
    
    #print(A.shape[1])
    x = cross_entropy_minimization(A)
    
    z = np.reshape(x, (81, 9))
    if np.linalg.norm(np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9) ) \
                      - np.reshape([int(c) for c in solu], (9,9)), np.inf) >0:
        pass
    else:
        #print("CORRECT")
        corr_cnt += 1

    if (i+1) % 20 == 0:
        end = time.time()
        print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(t=(end-start)/(i+1), corr=corr_cnt, all=i+1) )

end = time.time()
print("Aver Time: {t:6.2f} secs. Success rate: {corr} / {all} ".format(t=(end-start)/(i+1), corr=corr_cnt, all=i+1) )

