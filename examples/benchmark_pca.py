# Python impl of pca
# compare against rust impl
import numpy as np
import time
from sklearn.decomposition import PCA
from corrla_rs import rpca


def main():
    rng = np.random.RandomState(42)
    cov = np.random.uniform(0, 1, size=(12, 12))
    x_data = rng.multivariate_normal(mean=np.zeros(12), cov=cov, size=10000)
    n_components = 4

    # sklearn
    ti = time.perf_counter()
    sk_pca = PCA(n_components)
    sk_pca.fit(x_data)
    sk_pca_components = sk_pca.components_
    sk_pca_singular_vals = sk_pca.singular_values_
    tf = time.perf_counter()
    print("sklearn singular vals: ")
    print(sk_pca_singular_vals)
    print("sklearn elapsed time (s): ")
    print(tf - ti)

    # rust impl
    ti = time.perf_counter()
    rpca_singular_vals, rpca_components = rpca(x_data, n_components, 4, 6)
    tf = time.perf_counter()
    print("rsvd singular vals: ")
    print(rpca_singular_vals)
    print("rsvd elapsed time (s): ")
    print(tf - ti)


if __name__ == "__main__":
    main()
