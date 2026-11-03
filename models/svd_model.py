import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd

from utils.data_loader import load_matrix



def train_svd(matrix, k=50):
    matrix = matrix.to_numpy(dtype=np.float64)

    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or infinite values")

    U, sigma, Vt = svds(matrix, k=k)

    return U, sigma, Vt


if __name__ == "__main__":
    
    matrix = load_matrix()

    U, sigma, Vt = train_svd(matrix)

    print("U shape:", U.shape)
    print("Sigma shape:", sigma.shape)
    print("Vt shape:", Vt.shape)