import argparse
import numpy as np
from util import generate_binary_matrix
from formulation1 import qbmf_formulation1
from example import qbmf_direct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-proxy', action="store_true")
    args = parser.parse_args()
    m, n, r = 10, 8, 4

    # Generate matrix
    U, V = generate_binary_matrix(m, n, r=r)
    A = U @ np.transpose(V)
    density = np.count_nonzero(A)/(m*n)
    print("A is {}-by-{} and has a density of {}.".format(m, n, density))

    U1, V1 = qbmf_direct(A, r, proxy=args.proxy)
    U2, V2 = qbmf_formulation1(A, r, proxy=args.proxy)

    A1 = U1 @ np.transpose(V1) if (U1 is not None and V1 is not None) else None
    A2 = U2 @ np.transpose(V2) if (U2 is not None and V2 is not None) else None

    print(A1)
    print()
    print(A2)
    print()