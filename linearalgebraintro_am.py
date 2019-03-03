# linearalgebraintro_am.py

## Revision:	1.3
## Date:		03.03.2019 

import numpy as np
from numpy import pi

# NumPy is the fundamental package for scientific computing with Python.
# It contains -
# 		a powerful N-dimensional array object
#		sophisticated (braodcasting) functions
#		tools for integrating C/C++ and Fortran code
#		useful linear algebra, Fourier transform, and random number capabilities
# Source: www.numpy.org

print('Hello World! Welcome to this introduction to linear algebra in Python.\n')

### Vectors ###
# A vector is an array of numbers.
# A = [1, 2, 3]

A = np.array([1, 2, 3]) # We use NumPy module to intialize array.

### Matrices ###
# A matrix is a 2D array of similar numbers.
# M = [[ 1, 2, 3],[4, 5, 6]]

### Scalar ###
x = 3 # Scalars can be thought of as single numbers or even 1x1 matrix

# NumPy has a matrix function. However, as per official documentation, it will be removed in the future
M = np.array([(1, 2, 3),(4, 5, 6)]) # Always provide a list of numbers.


print('Vector')
print('A vector is an array of numbers.\nFor example - A: ' + str(A) + '\n')
print('Matrix')
print('A matrix is a 2D array of numbers.\nFor example - M: ' + str(M) + '\n')

# Elements of a vector and matrix
A[1] # 2
A[:] # array([1, 2, 3])
M # array([[1, 2, 3], [4, 5, 6]])
M[:,2] # array([3, 6])
M[1,2] # 6

# Initiailize with zeros, ones
B = np.zeros((2,2)) # zero matrix
C = np.ones((2,2), dtype=np.int16)
E = np.array([(1, 0, 0),(0, 1, 0),(0, 0, 1)]) # Identity matrix

print('Zero matrix: ' + str(B))

# Dimensions of vectors and matrices -> the number of axes (dimensions) of the array.
print('No. of dimensions of vector A: ' + str(A.ndim)) # 1
print('No. of dimensions of matrix M: ' + str(M.ndim) + '\n') # 2

# Shape of vectors and matrices -> the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension.
print('Shape of vector A: ' + str(A.shape)) # (3,)
print('Shape of matrix M: ' + str(M.shape) + '\n') # (2,3)

# Size of vectors and matrices -> the total number of elements of the array.
print('Size of vector A: ' + str(A.size)) # 3
print('Size of matrix M: ' + str(M.size) + '\n') # 6

# arange -> create numeric sequences in Python
S = np.arange(6)
print('S: ' + str(S) + '\n') # Syntax: arange(start, stop, step, dtype); Only stop is mandatory, default start is 0, default step is 1, dtype is inferred if not provided

# linspace -> use when building sequences with floating point arguments. It receives the number of elements as argument
Sf = np.linspace( 0, 2*pi, 100 )
print(Sf)
Sfsin = np.sin(Sf)
print(Sfsin)

# Reshape -> reshape an n1-dimensional array to n2-dimensional array. The size should be the same.
Sr = np.reshape(S,(2,3))
print('Reshaped array S to matrix of dimensions 2,3: ' + str(Sr) + '\n')

# Two matrices are equal if they have the same shape and elements
D = np.array([(1, 2, 3),(4, 5, 6)])
E = np.array([(1, 2, 4),(4, 5, 6)])

# Transpose of a matrix - returns a view of the array with axes transposed for n-D array
print('Transpose of matrix M is ' + str(M.T)) # [[1 4] [2 5] [3 6]]

# Determinant of a matrix - for a 2D array [[a b][c d]], it is ad - bc
F = np.array([(2,3), (4,5)])
print('Determinant of matrix F is '+ str(np.linalg.det(F)))

print('Matrices M and D are equal - ' + str(np.array_equal(M,D)))
print('Matrices M and E are equal - ' + str(np.array_equal(M,E)))

print('Reference:')
print('----------')
print('SciPy resource on Linear Algebra (https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)')
print('NumPy resource (http://www.numpy.org/)')
print('StatLect (https://www.statlect.com)')

