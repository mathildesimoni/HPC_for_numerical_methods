import time
import numpy as np
from functions import create_matrix_C, create_matrix_suitsparse, CGS_sequential, cholQR_sequential, TSQR_sequential
import sys

if __name__ == "__main__":
    matrix_fn = sys.argv[1]
    method = sys.argv[2]

    if len(sys.argv) != 3:
    	raise AssertionError("Please call the script the following way: QR_sequential.py matrix_creation_function method")
    print("Matrix: ", matrix_fn)
    print("Method: ", method)

    # Create the initial matrix W
    if matrix_fn == "C_sequential":
        m = 49984 
        n = 600
        W = create_matrix_C(m, n)
    elif matrix_fn == "D_sequential":
        file_name = "matrices/net125.mtx"
        nb_cols = 150
        W, m, n = create_matrix_suitsparse(file_name, nb_cols)
        W = W[:36672, :] # to make sure the number of rows divides 8, 16, 32, 64
        m = 36672
    else:
        raise ValueError("The initial matrix creation function is not defined!")

    wt = time.time()
    if method == "TSQR_sequential":
        Q, R = TSQR_sequential(W, m, n)
    elif method == "CGS_sequential":
        Q, R = CGS_sequential(W, m, n)
    elif method == "cholQR_sequential":
        Q, R = cholQR_sequential(W, m, n)
    else:
        raise ValueError("The QR factorization method is not defined!")

    # 3. RESULTS
    wt = time.time() - wt
    if method == "cholQR_sequential":
        D = W.T @ W
        print("WTW is SPD?: ", np.all(np.linalg.eigvals(D) > 0)) # check positive definiteness after the timer
        print("Condition number of WTW: ", np.linalg.cond(D))
    print("Size of W: ", W.shape)
    print("Condition number of W: ", np.linalg.cond(W))
    print("Size of Q:", Q.shape)
    print("Size of R:", R.shape)
    print("Time taken: ", wt)
    print("Orthogonality of Q: ", np.linalg.norm(np.eye(n) - Q.T@Q))
    print("Condition number of Q: ", np.linalg.cond(Q))
    if np.allclose(Q @ R, W, atol=1e-6): # small tolerance level
        print("Q@R = W: Success!")
    else:
        print("Q@R != W: Error!")











