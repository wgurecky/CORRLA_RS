"""
RSVD benchmarks. Ex, to run benchmark with 2 threads:

    OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 python benchmark_rsvd.py

NOTE:
If number of threads is not specified, both the python (numpy)
and rust methods will try to use all available cores
(or whatever the default OMP_NUM_THREADS is on the machine).
This can lead to unreliable performance on hyperthreaded cpus.
"""
import numpy as np
import corrla_rs as hrl


def power_iteration(A, omega, power_iter=8):
    Y = A @ omega
    for q in range(power_iter):
        # Q, _ = np.linalg.qr(Y)
        # Y = Q
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q


def rsvd(A, omega_rank=4, n_oversamples=4, power_iter=8):
    """
    Randomized SVD method from:

        N. Erichson, S. Voronin, S. Brunton, J. Kutz.
        Randomized Matrix Decompositions using R.
        https://arxiv.org/abs/1608.02148

    Args:
        A: matrix to decompose into A \approx Ur*Sr*Vr^T
            where the full svd is A=U*S*V^T, but Ur and Sr and Vr
            is only the first r cols of U, and first r singular
            values of S.
        omega_rank: desired rank of the rsvd
        power_iter: number of power iterations to perform
    """
    fat = False
    if A.shape[0] < A.shape[1]:
        fat = True
        A = A.T
    omega = np.random.randn(A.shape[1], omega_rank + n_oversamples)
    Q = power_iteration(A, omega, power_iter)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices=0)
    u = Q @ u_tilde
    if fat:
        return v[:omega_rank].T, s[:omega_rank], u[:, :omega_rank].T
    else:
        return u[:, :omega_rank], s[:omega_rank], v[:omega_rank]


if __name__ == "__main__":
    print("========= NUMPY MKL/BLAS CONFIG ========")
    np.show_config()
    print("========= ===================== ========")
    import time
    test_A = np.random.randn(100000, 1000)
    # test_A = np.random.randn(1000, 100000)
    ti = time.perf_counter()
    ur, sr, vr = rsvd(test_A, omega_rank=4,
                      n_oversamples=10, power_iter=8)
    tf = time.perf_counter()
    print("rsvd singular values:")
    print(sr)
    print("rsvd singular vectors:")
    print(ur)
    print("rsvd elapsed time (s): ")
    print(tf - ti)

#     ti = time.perf_counter()
#     u, s, v = np.linalg.svd(test_A, full_matrices=0)
#     tf = time.perf_counter()
#     print("svd singular values:")
#     print(s[:4])
#     print("svd singular vectors:")
#     print(u[:, :4])
#     print("svd elapsed time (s): ")
#     print(tf - ti)

    #sklearn randomized svd
    from sklearn.utils.extmath import randomized_svd
    ti = time.perf_counter()
    urb, srb, vrb = randomized_svd(test_A, n_components=4, n_iter=8)
    tf = time.perf_counter()
    print("sklearn rsvd singular values:")
    print(srb)
    print("sklearn rsvd singular vectors:")
    print(urb)
    print("sklearn rsvd v:")
    print(vrb)
    print("sklearn rsvd elapsed time (s): ")
    print(tf - ti)

    #rust randomized svd
    ti = time.perf_counter()
    urc, src, vrc = hrl.rsvd(test_A, 4, 8, 10)
    tf = time.perf_counter()
    print("rust rsvd singular values:")
    print(src)
    print("rust rsvd singular vectors:")
    print(urc)
    print("rust rsvd v:")
    print(vrc)
    print("rust rsvd elapsed time (s): ")
    print(tf - ti)
