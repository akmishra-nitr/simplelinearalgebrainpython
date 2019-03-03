# linearalgebraintro_am.py

## Revision: 1.2
## Date: 03.03.2019

import numpy as np
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

# NumPy has a matrix function. However, as per official documentation, it will be removed in the future
M = np.array([(1, 2, 3),(4, 5, 6)]) 

print('Vector')
print('A vector is an array of numbers.\nFor example - A: ' + str(A) + '\n')
print('Matrix')
print('A matrix is a 2D array of numbers.\nFor example - M: ' + str(M) + '\n')

print('Reference:')
print('----------')
print('SciPy resource on Linear Algebra (https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)')
print('NumPy resource (http://www.numpy.org/)')

