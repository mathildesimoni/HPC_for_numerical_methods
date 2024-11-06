from mpi4py import MPI
import numpy as np
from functions import is_power_of_two, create_matrix_C, create_matrix_suitsparse, CGS, cholQR, TSQR
import sys

if __name__ == "__main__":

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_fn = sys.argv[1]
    method = sys.argv[2]

    if rank == 0:
        if len(sys.argv) != 3:
        	raise AssertionError("Please call the script the following way: QR.py matrix_creation_function method")
        print("Matrix: ", matrix_fn)
        print("Method: ", method)

        if method == "TSQR" and not is_power_of_two(size):
            raise ValueError("The number of processes should be a power of 2!")

    W = None # Note that W is referenced as A in the report
    m = 0
    n = 0

    # Create the initial matrix W
    if rank == 0:
        if matrix_fn == "C":
            m = 49984 # not 50000 because the number rows should divide the number of processors
            n = 600 # 60 # 60
            W = create_matrix_C(m, n)
        elif matrix_fn == "D":
            file_name = "matrices/net125.mtx"
            nb_cols = 150
            W, m, n = create_matrix_suitsparse(file_name, nb_cols)
            W = W[:36672, :]
            m = 36672
        elif matrix_fn == "E":
            file_name = "matrices/ct20stif.mtx"
            nb_cols = 500
            W, m, n = create_matrix_suitsparse(file_name, nb_cols)
            W = W[:52288, :]
            m = 52288
        else:
            raise ValueError("The initial matrix creation function is not defined!")
    
    m = comm.bcast(m, root = 0)
    n = comm.bcast(n, root = 0)
    local_size = int(m/size)

    wt = MPI.Wtime() # We are going to time this
    if method == "TSQR":
        Q_local, R = TSQR(W, m, n, local_size, comm, rank, size)
    elif method == "CGS":
        Q_local, R = CGS(W, m, n, local_size, comm, rank, size)
    elif method == "cholQR":
        Q_local, R = cholQR(W, m, n, local_size, comm, rank, size)
    else:
        raise ValueError("The QR factorization method is not defined!")

    comm.Barrier()
    wt = MPI.Wtime() - wt

    # 3. RESULTS
    # here, we gather the local Qs to a big one in order to compute required quantities
    # note that this would not have been necessary if the only quantitiy to chekc was the loss of orthogonality
    # indeed, loss of orthogonality can be computed by first computing local matrix-matrix multiplication and then summing them with a reduce operation
    # however, we are also asked to compute the condition number of Q, which require the full matrix Q to be computed, so we do it anyway
    Q = None
    if rank == 0: 
        Q = np.zeros((m,n), dtype = 'd')
    comm.Gather(Q_local, Q, root = 0)

    # 3. RESULTS
    if(rank == 0):
        print("Size of W: ", W.shape)
        print("Condition number of W: ", np.linalg.cond(W))
        print("Size of Q:", Q.shape)
        print("Size of R:", R.shape)
        print("Time taken: ", wt)
        if (method == "CGS") and (size == 8): # only need to solve when P = 8 for numerical stability analysis
            orthogonalities = np.zeros(n, dtype = 'd')
            condition_numbers = np.zeros(n, dtype = 'd')
            for i in range(n):
                Q_tmp = Q[:, :i+1]
                orthogonalities[i] = np.linalg.norm(np.eye(i+1) - Q_tmp.T@Q_tmp)
                condition_numbers[i] = np.linalg.cond(Q_tmp)
            print("Orthogonality of Q: ", orthogonalities[-1])
            with open(method + "_" + matrix_fn + '_loss.npy', 'wb') as f:
                np.save(f, orthogonalities)
            with open(method + "_" + matrix_fn + '_cond.npy', 'wb') as f:
                np.save(f, condition_numbers)
        else:
            print("Orthogonality of Q: ", np.linalg.norm(np.eye(n) - Q.T@Q))
        print("Condition number of Q: ", np.linalg.cond(Q))
        if np.allclose(Q @ R, W, atol=1e-6): # small tolerance level
            print("Q@R = W: Success!")
        else:
            print("Q@R != W: Error!")











