import numpy as np
from numpy.linalg import norm
from mpi4py import MPI
import scipy
from scipy.io import mmread


def create_matrix_C(m, n):
	# as defined in the project description
    C = np.zeros((m, n), dtype='d')

    for i in range(m):
        x = (i - 1) / (m - 1)
        for j in range(n):
            mu = (j - 1) / (n - 1)
            C[i, j] = np.sin(10 * (mu + x)) / (np.cos(100 * (mu - x)) + 1.1)
    
    return C


# create matrix E and D
def create_matrix_suitsparse(file_name, cols):
	tmp = mmread(file_name).toarray()
	m, n = tmp.shape
	D = np.zeros((m, cols), dtype='d')
	D[:, :] = tmp[:, :cols]
	return D, m, cols


def CGS_sequential(W, m, n):
	R = np.zeros((n,n), dtype = 'd')
	Q = np.zeros((m,n), dtype = 'd')
	QT = np.zeros((n, m), dtype = 'd')

	R[0, 0] = np.linalg.norm(W[:, 0])
	Q[:, 0] = W[:, 0] / R[0, 0]
	QT[0, :] = Q[:, 0] # store in the transpose too (was tested to be a tiny bit faster)

	for j in range(1, n):
		R[:j, j] = QT[:j, :]@W[:, j]
		Q[:, j] = W[:, j] - Q[:, :j]@R[:j, j]
		R[j, j] = np.linalg.norm(Q[:, j])
		Q[:, j] = Q[:, j] / R[j, j]
		QT[j, :] = Q[:, j]

	return Q, R


def CGS(W, m, n, local_size, comm, rank, size):
	# specific initialization to CGS
	R = np.zeros((n,n), dtype = 'd') # need to declare this one on all processes
	norm_col_square = np.zeros(1)

	# Initialization of local matrices
	norm_local = np.zeros(1)
	W_local = np.zeros((local_size, n), dtype = 'd')
	Q_local = np.zeros((local_size, n), dtype = 'd')
	QT_local = np.zeros((n, local_size), dtype = 'd') # jut to avoid taking the transpose every time

	comm.Scatterv(W, W_local, root = 0)

	# norm square of the first column first
	norm_local = np.array(np.inner(W_local[:, 0], W_local[:, 0]))
	comm.Allreduce(norm_local, norm_col_square, op = MPI.SUM)
	R[0, 0] = np.array(np.sqrt(norm_col_square))

	Q_local[:, 0] = W_local[:, 0] / R[0, 0]
	QT_local[0, :] = Q_local[:, 0] # store in the transpose too

	comm.Barrier() # initialization finished

	for j in range(1, n):
		tmp_local = QT_local[:j, :]@W_local[:, j] # (j, local_size) x (local_size, 1) => (j, 1)
		
		R_tmp = np.zeros((j,), dtype = 'd')
		comm.Allreduce(tmp_local, R_tmp, op = MPI.SUM)
		R[:j, j] = R_tmp

		Q_local[:, j] = W_local[:, j] - Q_local[:, :j]@R[:j, j]

		norm_local = np.array(np.inner(Q_local[:, j], Q_local[:, j]))
		comm.Allreduce(norm_local, norm_col_square, op = MPI.SUM)
		R[j, j] = np.array(np.sqrt(norm_col_square))

		Q_local[:, j] = Q_local[:, j] / R[j, j]
		QT_local[j, :] = Q_local[:, j]

	return Q_local, R


def cholQR_sequential(W, m, n):
	G = W.T @ W
	R = np.linalg.cholesky(G).T # we want the upper triangular matrix
	Q = np.linalg.solve(R.T, W.T).T
	return Q, R


def cholQR(W, m, n, local_size, comm, rank, size):
	# specific initialization to CholQR
	G = np.zeros((n, n), dtype = 'd')

	# Distribute W on all processes
	W_local = np.zeros((local_size, n), dtype = 'd')
	comm.Scatterv(W, W_local, root = 0)

	# Compute G = W^T * W
	G_local = W_local.T @ W_local
	comm.Allreduce(G_local, G, op = MPI.SUM)

	# Compute cholesky factorization of G
	R = np.linalg.cholesky(G).T # we want the upper triangular matrix

	# Get Q with a linear solve locally and then gather
	Q_local = np.ravel(np.linalg.solve(R.T, W_local.T).T) 
	# Ravel is used so that gather is made the good way (row wise)

	return Q_local, R


def is_power_of_two(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1


def TSQR_sequential(W, m, n):
	Q, R = np.linalg.qr(W)
	return Q, R


# Optimized TSQR (Q gotten optimally as discussed in class)
def TSQR(W, m, n, local_size, comm, rank, size):
	R = None
	# Distribute W on all processes
	W_local = np.zeros((local_size, n), dtype = 'd')
	comm.Scatterv(W, W_local, root = 0)

	# At first step, compute local Householder QR
	Q_local, R_local = np.linalg.qr(W_local) # sequential QR with numpy

	# Store the Q factors generated at each depth level
	Q_factors = [Q_local]

	depth = int(np.log2(size))

	for k in range(depth):
	    I = int(rank)

	    # processes that need to exit the loop
	    # are the processes that has a neighbor I - 2**k in the previous loop
	    # also do not remove any process at the first iteration
	    if (k != 0) and ((I % (2**(k))) >= 2**(k-1)):
	        break

	    if (I % (2**(k+1))) < 2**k:
	        J = I + 2**k
	    else:
	        J = I - 2**k

	    if I > J:
	        comm.send(R_local, dest = J, tag = I+J) # this tag makes sure it is the same for both partners
	    else:
	        other_R_local = comm.recv(source = J, tag = I+J)
	        new_R = np.vstack((R_local, other_R_local))
	        Q_local, R_local = np.linalg.qr(new_R)
	        Q_factors.insert(0, Q_local)


	comm.Barrier() # make sure all have finished

	nb_Q_factors_local = len(Q_factors)

	# Now need to compute Q
	# Get Q in reverse order, starting from root to the leaves
	i_local = 0
	nb_Q_factors_local = len(Q_factors)
	if rank == 0:
	    R = R_local # R matrix was computed already, stored in process 0
	    Q_local = Q_factors[i_local] # Q is intialized to last Q_local
	    i_local += 1

	for k in range(depth-1, -1, -1): 
		# processes sending
		if nb_Q_factors_local > k + 1:
			I = int(rank)
			J = int(I+2**k)
			rhs = Q_local[:n, :]
			to_send = Q_local[n:, :]
			comm.send(to_send, dest = J)

		# processes receiving
		if nb_Q_factors_local == (k + 1):
			I = int(rank)
			J = int(I-2**k)
			rhs = np.zeros((n, n), dtype = 'd')
			rhs = comm.recv(source = J)

		# processes doing multiplications
		if nb_Q_factors_local >= k + 1:
			Q_local = Q_factors[i_local] @ rhs
			i_local += 1

	return Q_local, R
