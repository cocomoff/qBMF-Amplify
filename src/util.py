# This code is from the author's repository.
# See: https://github.com/OsmanMalik/Quantum-BMF


import numpy as np
import random
from scipy.stats import bernoulli

def generate_binary_matrix(m, n, r, pU=0.5, pV=0.5):
    """Generate factors for matrix with exact BMF rank.

    Generates an m-by-n binary matrix which has exact binary rank r. The 
    function returns the factors U and V for the matrix. Minor note: This 
    function is somewhat different from Algorithm 3 in the supplement of 
    [MUR+21].

    Args:
        m:
            Number of rows of binary matrix being created.
        n:
            Number of columns of the binary matrix being created.
        r:
            Target rank r.
        pU:
            Initial density of U matrix. Should be between 0 and 1.
        pV:
            Initial density of V matrix. Should be between 0 and 1.
    
    Returns:
        Returns the matrices U and V of size m-by-r and n-by-r, respectively, so
        that they form the matrix A = U @ np.transpose(V) which is binary and
        has binary rank r.

    References:
    [MUR+21]    O. A. Malik, H. Ushijima-Mwesigwa, A. Roy, A. Mandal, I. Ghosh. 
                Binary matrix factorization on special purpose hardware. PLOS 
                ONE 16(12): e0261250, 2021. DOI: 10.1371/journal.pone.0261250
    """

    # Initialize U and V randomly with predetermined average density
    U = bernoulli.rvs(pU, size=(m, r))
    V = bernoulli.rvs(pV, size=(n, r))
    A = U @ np.transpose(V)

    # Keep running the following until A is binary
    while (A>1).any():
        # Draw random entry in A which exceeds 1
        vecA = A.reshape(m*n)
        ridx = random.choice([k for k in range(m*n) if vecA[k]>1])
        row = ridx // n
        col = ridx % n

        # Determine entry in U or V to set to zero
        r_joint_idx = random.choice([k for k in range(r) if U[row, k]*V[col, k] > 0])
        
        # Set entry in either U or V to zero, depending on which is densest.
        if np.sum(U)/(m*r) > np.sum(V)/(n*r):  
            # U more dense than V, so eliminate nonzero in U
            U[row, r_joint_idx] = 0
        else:
            # V more dense than U, so eliminate nonzero in V
            V[col, r_joint_idx] = 0

        A = U @ np.transpose(V)

    return U, V