import argparse
import numpy as np
import matplotlib.pyplot as plt
from formulation1 import qbmf_formulation1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-proxy", action="store_true")
    parser.add_argument("-num", default=1, type=int)
    args = parser.parse_args()
    A = np.loadtxt("./data/target.csv")

    U1, V1 = qbmf_formulation1(
        A, 3, num_solves=args.num, proxy=args.proxy, verbose=False
    )
    A1 = U1 @ np.transpose(V1)

    Aimg = A * 255
    plt.style.use("ggplot")
    f = plt.figure(figsize=(8, 4))
    a = f.add_subplot(1, 2, 1)
    a.imshow(Aimg, cmap="Grays")
    a = f.add_subplot(1, 2, 2)
    a.imshow(A1 * 255, cmap="Grays")
    plt.tight_layout()
    plt.show()
    plt.close()

    print(A)
