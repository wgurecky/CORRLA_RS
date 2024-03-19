"""
Compares performance of corrla-rs DMD impl
and PyDMD on a DMDc problem
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pydmd
from pydmd.dmdc import DMDc

# corrla imports
from corrla_rs import PyDMDc


def build_snapshots():
    nx = 5000
    nt = 40
    x_points = np.linspace(0., 9.5, nx, dtype=np.float64)
    t_points = np.linspace(0., 9.75, nt, dtype=np.float64)

    # control input
    u_seq = np.ones((1, len(t_points)), dtype=np.float64)
    def u_fn(t):
        return np.exp(0.2*t)+0.8*t
    for i, t in enumerate(t_points):
        u_seq[0, i] = u_fn(t)

    # response
    p_snapshots = []
    p_fn = lambda x, t: (np.sin(0.2*x+0.2*t)**2.)*u_fn(t) + np.random.rand(len(x))*1e-1
    for t in t_points:
        p_snapshot = p_fn(x_points, t)
        p_snapshots.append(p_snapshot)
    p_snapshots = np.asarray(p_snapshots).T
    plot_snapshots(x_points, t_points, p_snapshots, title="Original Data")

    return p_snapshots, u_seq, t_points, x_points


def plot_snapshots(x_points, t_points, p_snapshots, fig_f="pydmdc_snapshots.png", title=""):
    plt.figure()
    colors = plt.cm.jet(np.linspace(0,1,len(t_points)))
    for i, (t, p_snapshot) in enumerate(zip(t_points, p_snapshots.T)):
        plt.plot(x_points, p_snapshot, label="t=" + str(t), color=colors[i])
    plt.legend(ncol=3, fontsize=8)
    plt.title(title)
    plt.ylim(0, 12)
    plt.grid(ls="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(fig_f)
    plt.close()


def predict_dmdc(pydmdc_model: DMDc, x0, u_seq):
    """
    Constructs model of the form
    x_t+1 = A x_t + B u_t

    Args:
        x_0: shape=(n_features, 1)
        u_seq: 2d numpy array with colums representing
            control var state.  shape=(n_contorls, n_times)
    """
    # ensure single initial state
    assert x0.shape[1] == 1
    assert len(u_seq.shape) == 2
    assert len(x0.shape) == 2

    # extract approx \tilde A and \tilde B lin operators
    eigs = pydmdc_model.eigs
    B_til = pydmdc_model._B
    A_til =  np.linalg.multi_dot(
        [pydmdc_model.modes, np.diag(eigs), np.linalg.pinv(pydmdc_model.modes)]
    )

    predicted_x = [x0,]
    for i, u_col in enumerate(u_seq.T):
        u_col = np.asarray([u_col]).T
        next_x = A_til @ predicted_x[i] + B_til @ u_col
        predicted_x.append(next_x.real)
    return np.hstack(predicted_x)

def fit_pydmd():
    p_snapshots, u_seq, t_points, x_s = build_snapshots()
    n_modes = 12
    ti = time.time()
    pydmdc_model = DMDc(svd_rank=n_modes, svd_rank_omega=n_modes)
    pydmdc_model.fit(p_snapshots, u_seq[:, 1:])
    tf = time.time()
    print("PyDMD Fit time: ", tf - ti)

    # check reduced rank ops
    #print("pydmdc _A til: ", pydmdc_model._Atilde._Atilde)
    #print("pydmdc _B til: ", pydmdc_model._B)
    # print("pydmdc _basis: ", pydmdc_model._basis)
    # check eigs
    print("pydmdc eigs: ", pydmdc_model.eigs)
    # check modes
    #print("pydmdc modes: ", pydmdc_model.modes)

    # forecast
    predicted = predict_dmdc(pydmdc_model,
                             p_snapshots[:, 0:1],
                             u_seq[:, 1:]
                             )

    print("pydmdc predicted: ", predicted[:, 20])
    print("pydmdc expected: ", p_snapshots[:, 20])
    eval_t = t_points[1:]
    plot_snapshots(x_s, eval_t, predicted, fig_f="pydmdc_dmd_predictions.png",
                   title="PyDMD Predicted")
    mae_list = []
    for i in range(len(eval_t)):
        diffs = predicted[:, i] - p_snapshots[:, i+1]
        mae = np.mean(np.abs(diffs))
        mae_list.append(mae)
    plt.figure()
    plt.plot(eval_t, mae_list)
    plt.title("Other PyDMD DMDc code model \n Mean Absolute Diffs: | DMDc model - truth |")
    plt.grid(ls="--")
    plt.ylabel("Mean Abs Error")
    plt.xlabel("forecast time")
    plt.savefig("dmd_pydmdc_errors.png")
    plt.close()

def fit_corrla_dmd():
    p_snapshots, u_seq, t_points, x_s = build_snapshots()
    n_modes = 12
    n_iters = 10
    ti = time.time()
    rust_dmdc = PyDMDc(p_snapshots, u_seq, n_modes, n_iters)
    tf = time.time()
    print("Corrla DMDc Fit time: ", tf - ti)

    x0 = p_snapshots[:, 0:1]
    u_s = u_seq[:, 1:]
    predicted = rust_dmdc.predict(x0, u_s)
    print("corrla dmdc predicted: ", predicted[:, 20])
    print("corrla dmdc expected: ", p_snapshots[:, 20])

    eval_t = t_points[1:]
    plot_snapshots(x_s, eval_t, predicted, fig_f="corrla_dmd_predictions.png",
                   title="Corrla DMDc Predicted")
    mae_list = []
    for i in range(len(eval_t)):
        diffs = predicted[:, i] - p_snapshots[:, i+1]
        mae = np.mean(np.abs(diffs))
        mae_list.append(mae)
    plt.figure()
    plt.plot(eval_t, mae_list)
    plt.title("My DMDc code model \n Mean Absolute Diffs: | DMDc model - truth |")
    plt.grid(ls="--")
    plt.ylabel("Mean Abs Error")
    plt.xlabel("forecast time")
    plt.savefig("dmd_corrla_errors.png")
    plt.close()


if __name__ == "__main__":
    fit_pydmd()
    fit_corrla_dmd()
