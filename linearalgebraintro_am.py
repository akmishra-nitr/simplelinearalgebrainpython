# linearalgebraintro_am.py
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
# A vector is an array of similar data types.
# A = [1.0, 2.0, 3.0]

A = np.array([1, 2, 3]) # We use numpy module to intialize array

print("A vector is an array of similar data types.\nFor example - A: " + str(A) + "\n")

print('Reference:')
print('----------')
print('SciPy resource on Linear Algebra (https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html)')
print('NumPy resource (http://www.numpy.org/)')

