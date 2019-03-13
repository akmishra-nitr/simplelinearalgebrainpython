# solvinglinearequation_am.py

## Revision:	1.0
## Date:		13.03.2019 

import numpy as np
from numpy.linalg import inv

print('Hello World! Welcome to this exercise on solving a simple linear equation using matrices in Python.\n')

# Solving a linear equation for output. Example - A factory produces two products that use different quantities of two raw materials x1, x2. Price of the raw materials are included in the matrix X. The matrix A represents the relative quantities of x1, and x2 used to produce two Products P1 and P2. So, the matrix P represents the cost of producing P1 and P2.

X = np.array([3, 2])
A = np.array([(1, 2),(4, 5)])
P = np.dot(A,X)
B = np.eye(2) # 2D identity matrix

print('Product cost matrix: ' + str(P)) # Product cost matrix: [ 7 22]

# Inverse of a matrix
Ainv = inv(A)
np.allclose(np.dot(A, Ainv), np.eye(2)) # Test if the two arrays are element-wise equal within a tolerance

# The factory can be solved the other way. Let us say, we want to have price of 7 for P1 and 22 for P2. What should be the price of x1 and x2?

# Ax = P
# (AinvA)x = AinvP
# Ix = AinvP
X2 = np.dot(Ainv,P) # [3., 2.]

print('Reference:')
print('----------')
print('NumPy resource (http://www.numpy.org/)')
print('StatLect (https://www.statlect.com)')

