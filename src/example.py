import numpy as np
import argparse
from API_KEY import AMPLIFY_KEY as TOKEN, PROXY_STR
from amplify import FixstarsClient, VariableGenerator
from amplify import einsum, solve
from util import generate_binary_matrix

def qbmf_direct(
    A: np.ndarray,
    r: int = 2,
    proxy: bool = False,
) -> None:
    m, n = A.shape
    print(m, n)
    print(A)
    print()
    print(f"Token: {TOKEN}")
    print(f"Rank: {r}")

    # Solve Formulation 0 via Fixstars Amplify
    # 変数
    gen = VariableGenerator()
    U = gen.array("Binary", shape=(m, r))
    V = gen.array("Binary", shape=(n, r))

    # 目的関数
    froA = np.linalg.norm(A, ord="fro") ** 2
    term2 = einsum("ij,ik,jk->", A, U, V)
    term3 = einsum("ik,il,jk,jl->", U, U, V, V)
    model = froA - 2 * term2 + term3

    # 解く
    if proxy:
        client = FixstarsClient(proxy=PROXY_STR)
    else:
        client = FixstarsClient()
    client.token = TOKEN
    client.parameters.timeout = 1000

    ## 解を求める
    result = solve(model, client, num_solves=1)
    if len(result) > 0:
        U = U.evaluate(result.best.values)
        V = V.evaluate(result.best.values)
        print("U=")
        print(U)
        print()
        print("V=")
        print(V)
        print()
        print("UV^\\top=")
        print(U.dot(V.transpose()))
        print()
        print("|A - UV^\\top|=")
        print(A - U.dot(V.transpose()))
        print()
        return U, V

    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-proxy', action="store_true")
    args = parser.parse_args()
    # A = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]])
    # r = 2
    
    m, n, r = 10, 8, 4
    # Generate matrix
    U, V = generate_binary_matrix(m, n, r=r)
    A = U @ np.transpose(V)
    density = np.count_nonzero(A)/(m*n)
    print("A is {}-by-{} and has a density of {}.".format(m, n, density))
    qbmf_direct(A, r, proxy=args.proxy)