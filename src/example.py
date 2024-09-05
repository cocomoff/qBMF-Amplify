import numpy as np
import argparse
from API_KEY import AMPLIFY_KEY as TOKEN, PROXY_STR
from amplify import FixstarsClient, VariableGenerator
from amplify import einsum, sum, equal_to, less_equal, solve


def qbmf_formulation1(
    A: np.ndarray,
    r: int = 2,
    proxy: bool = False,
) -> None:
    m, n = A.shape
    print(m, n)
    print(A)
    print()

    lam = 2.1 * r * np.linalg.norm(A, ord="fro") ** 2
    print(f"Token: {TOKEN}")
    print(f"Rank: {r}")
    print(f"Penalty: {lam:>.3f}")

    # Solve Formulation 0 via Fixstars Amplify
    # 変数
    gen = VariableGenerator()
    U = gen.array("Binary", shape=(n, m))
    V = gen.array("Binary", shape=(n, m))
    W = gen.array("Binary", shape=(n, m, r))

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
        print(result.best)
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

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-proxy', action="store_true")
    args = parser.parse_args()
    A = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]])
    r = 2
    qbmf_formulation1(A, r, proxy=args.proxy)
