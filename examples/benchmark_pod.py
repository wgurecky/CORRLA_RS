# Rust Proper Orthogonal Decomposition demo and compare against python impl
import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RBFInterpolator

# corrla imports
from corrla_rs import PyPodI


class PODInterpModel(object):
    """
    POD with interpolation implementation.
    Author: William Gurecky
    From open source package: code.ornl.gov/dt-hydro/dt-hydro-cfd
    License for this class: Apache v2.
        See code.ornl.gov/dt-hydro/dt-hydro-cfd

    Args:
        max_modes: Maximum number of POD modes to compute and store.
            fewer modes are less accurate, but cost lest time to compute and store.
        interp_method: Method used to interpolate weights between known
            snapshot locations. Either "linear" or "spline" or "nn"
    """
    def __init__(self, interp_method='linear', max_modes=10, pod_method="svd"):
        self.interp_method = interp_method
        self.max_modes = max_modes
        self.n_t_dim = 0
        self.n_snap = 0
        self.pod_method = pod_method
        self.pod_modes = None
        self.pod_weights = None
        self.pod_w_interp_fs = []

    def fit(self, snap_x, t=None):
        """
        Fit the POD model

        Args:
            snap_x:  Flattened snapshots of scalar field field.
                Has shape (n_snapshots, n_space_sample_points)
            t: independant variable values that go with each snapshot.
                Usually time points.
        """
        assert len(snap_x.shape) == 2
        if t is None:
            # Default independant var (exploratory feature array)
            t = np.array([np.linspace(0, 1, len(snap_x))])
        t = np.reshape(t, [-1, 1])
        assert len(t.shape) == 2
        # dimension of independant vars
        self.n_t_dim = t.shape[1]
        self.n_snap = len(snap_x)
        assert self.n_snap == t.shape[0]
        # Compute pod modes
        if self.pod_method == "eig":
            pod_modes = self._fit_eigv(snap_x, t)
        else:
            pod_modes = self._fit_svd(snap_x, t)
        # Compute pod weights
        opt_pod_weights = self._fit_pod_weights(snap_x, pod_modes, t)
        self.pod_modes = pod_modes
        # for each snapshot, holds optimal pod weights that best
        # reconstruct each snapshot.
        self.pod_weights = opt_pod_weights
        # Fit interpolants that predict pod weights given indep var val
        self.pod_w_interp_fs = self._fit_pod_interp(opt_pod_weights, t)

    def _fit_pod_weights(self, snap_x, pod_modes, t, *args, **kwargs):
        """
        Fit the POD mode weights
        """
        opt_pod_weights = []
        for x_s in snap_x:
            # alpha_w_0 = np.ones(pod_modes.shape[1])
            # solve the least squares problem:
            # alpha_w_opt = argmin_alpha_w ||np.matmul(pod_modes, alpha_w) - x_s||
            alpha_w_opt, _r, _rk, _sv = np.linalg.lstsq(pod_modes, x_s)
            opt_pod_weights.append(alpha_w_opt)
        return np.asarray(opt_pod_weights)

    def _fit_pod_interp(self, opt_pod_weights, t, *args, **kwargs):
        pod_w_interp_fs = []
        for i, pod_ws in enumerate(opt_pod_weights.T):
            if self.n_t_dim == 1 and self.interp_method != "rbf":
                pod_w_interp_fs.append(interp1d(t.flatten(), pod_ws, kind=self.interp_method))
            else:
                # case for N-D intepolation of POD weights
                pod_w_interp_fs.append(RBFInterpolator(t, pod_ws, degree=1))
        return pod_w_interp_fs

    def _fit_eigv(self, snap_x, t, *args, **kwargs):
        """
        POD by eigen decomposition of correlation matrix.
        """
        # compute covar matrix of snapshots
        cov_mat = (1. / (self.n_snap - 1)) * np.matmul(snap_x.T, snap_x)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        modes = np.real(eig_vecs)[:, 0:self.max_modes]
        return modes

    def _fit_svd(self, snap_x, t, *args, **kwargs):
        """
        POD by SVD.  Faster than POD by eigen decomp.
        May be slightly less accurate.
        """
        from sklearn.utils.extmath import randomized_svd
        U, Sigma, VT = randomized_svd(snap_x, n_components=self.max_modes)
        modes = VT.T
        return modes

    def predict(self, t, *args, **kwargs):
        """
        Predicts field at point t using POD reconstruction
        """
        pod_weights_hat = np.zeros(self.pod_modes.shape[1])
        for i, w_fn in enumerate(self.pod_w_interp_fs):
            pod_weights_hat[i] = w_fn(t)
        return np.matmul(self.pod_modes, pod_weights_hat)


def example_pod(n_pod_modes=4):
    """
    Example using POD with interpolation to create model
    of 1D pressure field data as a fn on vane angle.
    Uses simple, synthetic data for testing.

    Args:
        n_pod_modes: Number of eigen modes to retain in the pod model.
            A truncated SVD is done to find the dominate modes.
            A larger number of retained modes is more accurate but slower.
    """
    # Construct synthetic data
    n_avail_snapshots = 20
    # indipendent variable array, represents vane angle open fraction
    grid_u = np.linspace(0., 1., n_avail_snapshots)
    # spatial grid (could by 3D, but here is 1D)
    # note: if 2D or 3D field data, ensure to flatten data arrays
    # before applying POD.
    grid_x = np.linspace(0., 10., 5000)
    # test pressure profiles, dependant var snapshot storage
    test_p_snaps = []
    test_p_t = []
    # Dummy pressure field parameters. Only used to generate synthetic data
    p_decay = 0.1
    p_wave_sd = 2.0
    p_wave_centers = np.linspace(0., 10., n_avail_snapshots)

    p_wave_centers_fn = lambda u: \
            ((np.max(grid_x) - np.min(grid_x)) / (np.max(grid_u) - np.min(grid_u))) * u
    def gen_p_profile(u):
        """
        Args:
            u: relative vane angle in [0, 1]
        """
        p_decay = 0.1
        p_wave_sd = 2.0
        p_scale = np.exp(-p_decay * u)
        p_center = p_wave_centers_fn(u)
        p_profile = norm(loc=p_center, scale=p_wave_sd * (1. + u)).pdf(grid_x) * p_scale
        return p_profile

    plt.figure()
    for i, u in enumerate(grid_u):
        # generate example pressure profiles
        p_scale = np.exp(-p_decay * u)
        # center of synthetic pressure wave
        p_center = p_wave_centers[i]
        p_center_b = p_wave_centers_fn(u)
        print(p_center, p_center_b)
        p_profile = norm(loc=p_center, scale=p_wave_sd * (1. + u)).pdf(grid_x) * p_scale
        test_p_snaps.append(p_profile)
        test_p_t.append(u)
        plt.plot(grid_x, p_profile, label="vane u: %0.2f" % u)
    plt.ylabel("Pressure [arbitrary scale]")
    plt.xlabel("Space [m]")
    plt.grid(ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1.1))
    plt.tight_layout()
    plt.savefig("pod_example_a_p_snaps.png")
    plt.close()
    # Convert input data to numpy arrays
    test_p_snaps = np.asarray(test_p_snaps)
    test_p_t = np.asarray(test_p_t)

    # Python POD model fitting
    ti = time.time()
    pod_model = PODInterpModel(max_modes=n_pod_modes, pod_method="svd")
    pod_model.fit(test_p_snaps, test_p_t)
    tf = time.time()
    print("Python POD fit time: %0.2e (s)" % (tf-ti))

    # Rust POD model fitting
    ti = time.time()
    corrla_pod = PyPodI(test_p_snaps, test_p_t.reshape((-1, 1)), n_pod_modes)
    tf = time.time()
    print("Corrla POD fit time: %0.2e (s)" % (tf-ti))

    # Python POD prediction
    u_test = (grid_u[3] + grid_u[4]) / 2.
    known_p_test = gen_p_profile(u_test)
    ti = time.time()
    py_p_test = pod_model.predict(u_test)
    tf = time.time()
    print("Python POD predict time: %0.2e (s)" % (tf-ti))

    # Rust POD prediction
    ti = time.time()
    corrla_p_test = corrla_pod.predict(u_test.reshape((1, 1)))
    tf = time.time()
    print("Rust POD predict time: %0.2e (s)" % (tf-ti))

    plt.figure()
    plt.plot(grid_x, py_p_test.flatten(), label="Python")
    plt.plot(grid_x, corrla_p_test.flatten(), ls="--", label="Rust Corrla")
    plt.legend()
    plt.ylabel("Pressure [arbitrary scale]")
    plt.xlabel("Space [m]")
    plt.grid(ls="--")
    plt.savefig("pod_example_pred_rs_py.png")
    plt.close()

if __name__ == "__main__":
    example_pod()
