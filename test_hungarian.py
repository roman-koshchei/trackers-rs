import numpy as np
from scipy.optimize import linear_sum_assignment

# Test case 1: Constant cost matrix (should return identity)
C1 = np.array([[4.0, 1.0, 3.0],
               [2.0, 0.0, 5.0],
               [3.0, 2.0, 2.0]])
row_ind1, col_ind1 = linear_sum_assignment(C1)
print("Test 1 - 3x3 matrix:")
print("Cost matrix:")
print(C1)
print("Row indices:", row_ind1)
print("Col indices:", col_ind1)
print("Cost:", C1[row_ind1, col_ind1].sum())
print()

# Test case 2: Constant cost matrix (should return identity)
C2 = np.ones((4, 4))
row_ind2, col_ind2 = linear_sum_assignment(C2)
print("Test 2 - Constant 4x4 matrix:")
print("Cost matrix:")
print(C2)
print("Row indices:", row_ind2)
print("Col indices:", col_ind2)
print("Cost:", C2[row_ind2, col_ind2].sum())
print()

# Test case 3: Rectangular (more rows than columns)
C3 = np.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]])
row_ind3, col_ind3 = linear_sum_assignment(C3)
print("Test 3 - 4x3 matrix:")
print("Cost matrix:")
print(C3)
print("Row indices:", row_ind3)
print("Col indices:", col_ind3)
print("Cost:", C3[row_ind3, col_ind3].sum())
print()

# Test case 4: Rectangular (more columns than rows)
C4 = np.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0]])
row_ind4, col_ind4 = linear_sum_assignment(C4)
print("Test 4 - 3x4 matrix:")
print("Cost matrix:")
print(C4)
print("Row indices:", row_ind4)
print("Col indices:", col_ind4)
print("Cost:", C4[row_ind4, col_ind4].sum())
