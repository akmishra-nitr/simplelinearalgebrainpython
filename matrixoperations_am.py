# linearalgebraintro_am.py

## Revision:	1.0
## Date:		10.03.2019 

import numpy as np
from numpy import pi

print('Hello World! Welcome to this exercise on arithmetic operations in Python.\n')


x = 3
A = np.array([1, 2, 3])
M = np.array([(1, 2, 3),(4, 5, 6)])
B = np.array([5, 6, 7])


# Matrix addition and subtraction
C = np.add(A,M) # C = [[2 4 6][5 7 9]]
D = np.subtract(A,M) # D = [[ 0  0  0][-3 -3 -3]]
E = np.add(A,x) # E = [4,5,6]

print('Addition of Vector A and Matrix M resulting in matrix C: '+str(C)) # due to broadcasting
print('Subtraction of Matrix M from Vector A resulting in matrix D: '+str(D)) # due to broadcasting
print('Addition of Vector A and scalar x resulting in matrix E: '+str(E)) # due to broadcasting



# Matrix multiplication
F = np.dot(x,4) # F = 12
G = np.dot(A,x) # G = [3, 6, 9]; Same as A*x
H = np.dot(A,B) # H = 38

L = np.array([[1, 0],[1, 1]])
M = np.array([[4, 1],[2, 3]])

P = np.dot(L,M) # P = [[4, 1],[6, 4]]; Same as L@M; matrix product or inner product(for vectors)
Q = L*M # Q = [[4, 0],[2, 3]]; element-wise product
np.dot(L,A) # Error as shapes (2,2) and (3,) not aligned
N = np.reshape(A,(3,1)) # reshaping A to matrix of shape (3,1)
np.dot(L,A) # Error again as shapes (2,2) and (3,1) not aligned
O = np.array([1, 2])
R = np.dot(L,O) # [1, 3]

# Properties of matrix multiplication
T = np.array([[1, 0],[1, 1]])
U = np.array([2, 3])
V = np.array([4, 5])
X = np.array([[4, 5],[6, 7]])

print('Matrix multiplication is distributive with respect to matrix addition, that is,')
print('T(U+V) = TU + TV, when the multiplications and additions are meaningfully defined')
print('T(U+V) :' + str(np.dot(T, U+V))) # T(U+V) :[ 6 14]
print('TU+TV :' + str(np.dot(T,U)+ np.dot(T,V))) # TU+TV :[ 6 14]


print('Matrix multiplication is associative, that is,')
print('T(XV) = (TX)V, when the multiplications are meaningfully defined')
print('T(XV) :' + str(np.dot(T, np.dot(X,V)))) # T(XV) :[ 41 100]
print('(TX)V :' + str(np.dot(np.dot(T,X),V))) # (TX)V :[ 41 100]




print('Reference:')
print('----------')
print('NumPy resource (http://www.numpy.org/)')
print('StatLect (https://www.statlect.com)')

