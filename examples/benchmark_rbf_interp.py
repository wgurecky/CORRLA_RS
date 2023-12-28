import numpy as np
import time
import matplotlib.pyplot as plt
from corrla_rs import PyRbfInterp

def main():
    # create 2d function to interpolate
    f_xx = lambda x1, x2: np.sin(x1) + np.sin(x2)
    # eval fn at random points
    x1 = np.random.uniform(-np.pi, np.pi, 50)
    x2 = np.random.uniform(-np.pi, np.pi, 50)
    xx1, xx2 = np.meshgrid(x1, x2)
    y_xx = f_xx(xx1.flatten(), xx2.flatten())

    # plot target function
    plt.figure()
    plt.tricontourf(xx1.flatten(), xx2.flatten(), y_xx)
    plt.scatter(xx1.flatten(), xx2.flatten(), c='r', s=3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.savefig("rbf_interp_targ_f.png")
    plt.close()

    # init the rbf interpolator
    ti = time.perf_counter()
    corrla_rbf = PyRbfInterp(2, 1.0, 2, 1)
    x_support = np.asarray((xx1.flatten(), xx2.flatten())).T
    y_support = np.asarray((y_xx.flatten(),)).T
    corrla_rbf.fit(x_support, y_support)
    tf = time.perf_counter()
    print("fit elapsed time (s): ")
    print(tf - ti)

    # exact result
    x1_eval = np.linspace(-np.pi, np.pi, 20)
    x2_eval = np.linspace(-np.pi, np.pi, 20)
    xx1_eval, xx2_eval = np.meshgrid(x1_eval, x2_eval)
    y_xx_expected = f_xx(xx1_eval.flatten(), xx2_eval.flatten())

    # interpolate
    x_test = np.asarray((xx1_eval.flatten(), xx2_eval.flatten())).T
    ti = time.perf_counter()
    y_test = corrla_rbf.predict(x_test)
    tf = time.perf_counter()
    print("predict elapsed time (s): ")
    print(tf - ti)

    plt.figure()
    plt.contourf(xx1_eval, xx2_eval, y_test.reshape(xx1_eval.shape))
    plt.scatter(xx1_eval.flatten(), xx2_eval.flatten(), c='r', s=3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.savefig("rbf_interp_predict.png")
    plt.close()


if __name__ == "__main__":
    main()
