import numpy as np
import time
import matplotlib.pyplot as plt


def py_cs_dirichlet_sample(
        bounds, n_samples: int,
        max_zshots: int=500,
        chunk_size: int=2000000,
        c_scale=1.0):
    """
    Draws samples, x_i such that
        \sum_i x_i = C
        where C = 1 by default
        and
        lb_i <= x_i <= ub_i

    Author: William Gurecky
    Date: Jan 1 2024

    Args:
        bounds: list of tuples: [(lb_i, ub_i), ... ]
        n_samples: number of samples desired
        max_zshots: number of shots with each each 'shot' having chunk_size
            number of samples from the Dirichlet dist.
            Could in parallelize over shots.
        chunk_size: number of samples per shot
    """
    out_samples = []
    alphas = [1.0] * len(bounds)
    for zshot in range(max_zshots):
        print("Shot N= %d", zshot)
        # unconstrained samples, uniform in z
        uncstr_z_samples = np.random.dirichlet(alphas, size=chunk_size) * c_scale

        # select samples that meet bound constraints
        bool_mask = np.ones(len(uncstr_z_samples), dtype=bool)
        for i, bound in enumerate(bounds):
            tmp_mask = (bound[0] < uncstr_z_samples[:, i]) & (uncstr_z_samples[:, i] < bound[1])
            bool_mask = (bool_mask & tmp_mask)
        constr_z_samples = uncstr_z_samples[bool_mask]

        # append feasible samples to the output
        for s in constr_z_samples:
            out_samples.append(s)
        print("N valid samples= %d", len(out_samples))

        # done if number of valid samples meets or exceeds number requested
        if len(out_samples) >= n_samples:
            out_samples = np.asarray(out_samples)
            out_samples = out_samples[0:n_samples, :]
            break
    return out_samples


def hybrid_mcmc_dirichlet_sample(bounds, n_samples):
    """
    This method demos the hybrid constrained mcmc-dirichlet sampler.
    CORRLA-RS can use the standard dirichlet sampler w rejection sampling
    to generate _seed_ samples for a follow-up mcmc sampling run that
    more efficiently crawls a linear, constrained high dimensional surface.
    This is because, the MCMC sampler uses differential-evolution moves
    which constructs newly proposed states by adding the vector-difference
    between two random chains in the pool to a current chain position.
    This constrains the proposed new points to stay within-the-hyperplane
    and tends to stay within the feasible region.
    """
    from corrla_rs import cs_mcmc_dirichlet_sample
    # Bias the dirichlet samples into the tails (corners).
    # This is advantagous when
    np_alphas = np.asarray([0.6] * len(bounds))
    n_seed_samples = 12
    # DEMC move shrink factor
    gamma = 0.8
    # DEMC move jitter factor
    var_epsilon = 1e-12
    ti = time.time()
    samples, ar = cs_mcmc_dirichlet_sample(
            bounds, n_samples, n_seed_samples,
            500, 1000000, 1.0, np_alphas,
            gamma, var_epsilon)
    tf = time.time()
    print("CORRLA-RS mcmc-dirichlet hybrid sample time: %0.4e" % (tf-ti))
    print("samples")
    print(samples)
    print("avg mcmc acceptance rate")
    print(ar)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(samples[:, 0], samples[:, 1], samples[:, 2], s=2, alpha=0.5)
    ax.set_title("CORRLA-RS hybrid mcmc-dirichlet sampler")
    ax.set_xlabel("u234")
    ax.set_ylabel("u235")
    ax.set_zlabel("u238")
    plt.show()


def dirichlet_sample_bench(bounds, n_samples):
    """
    Compares a pure python dirichlet sampler with the rust based version
    in CORRLA-RS
    """
    # python impl
    ti = time.time()
    samples = py_cs_dirichlet_sample(bounds, n_samples)
    tf = time.time()
    print("Python sample time: %0.4e" % (tf-ti))
    print(samples)
    print(len(samples))
    # ensure sum constraint was met for each sample
    for sample in samples:
        assert np.isclose(np.sum(sample), 1.0, atol=1e-12)

    # bench rust impl
    try:
        # corrla imports
        from corrla_rs import cs_dirichlet_sample

        ti = time.time()
        alphas = [1.0] * len(bounds)
        np_alphas = np.asarray(alphas)
        samples = cs_dirichlet_sample(bounds, n_samples, 500, 1000000, 1.0, np_alphas)
        tf = time.time()
        print("Rust Corrla sample time: %0.4e" % (tf-ti))
        print(len(samples))
        # ensure sum constraint was met for each sample
        for sample in samples:
            assert np.isclose(np.sum(sample), 1.0, atol=1e-12)
    except ImportError:
        pass

    # Plots
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5)
    plt.xlabel("u234")
    plt.ylabel("u235")
    plt.savefig("constr_dirichlet_u234_u235.png")
    plt.close()
    plt.figure()
    plt.scatter(samples[:, 1], samples[:, 2], s=3, alpha=0.5)
    plt.xlabel("u235")
    plt.ylabel("u238")
    plt.savefig("constr_dirichlet_u235_u238.png")
    plt.close()

    plt.figure()
    plt.hist(samples[:, 0])
    plt.xlabel("u234")
    plt.savefig("constr_dirichlet_margin_u234.png")
    plt.close()
    plt.figure()
    plt.hist(samples[:, 1])
    plt.xlabel("u235")
    plt.savefig("constr_dirichlet_margin_u235.png")
    plt.close()
    plt.figure()
    plt.hist(samples[:, 2])
    plt.xlabel("u238")
    plt.savefig("constr_dirichlet_margin_u238.png")
    plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(samples[:, 0], samples[:, 1], samples[:, 2], s=2, alpha=0.5)
    ax.set_xlabel("u234")
    ax.set_ylabel("u235")
    ax.set_zlabel("u238")
    plt.savefig("constr_dirichlet_u234_u235_u238_surface.png")
    plt.close()

if __name__ == "__main__":
    bounds = np.asarray((
            # U234
            (0, 0.0026),
            # U235
            (0.1955, 0.1995),
            # U238
            (0.80, 0.825),
            ))
    n_samples = 3000

    # demo the hybrid mcmc-dirichlet sampler
    hybrid_mcmc_dirichlet_sample(bounds, n_samples)

    # run the python vs rust bench for standard dirichlet rejection sampler
    dirichlet_sample_bench(bounds, n_samples)
