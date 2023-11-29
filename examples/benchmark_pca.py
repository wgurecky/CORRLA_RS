# Python impl of pca
# compare against rust impl
import numpy as np
from sklearn.decomposition import PCA
from corrla_rs import rpca


main():
    cov
    rng = np.random.RandomState(0)
    cov = np.array([
            [],
            [],
            [],
            [],
            ])
    data = rng.multivariate_normal(mean=np.zeros(12), cov=cov, size=1000)
    n_components = 4
    sk_pca = PCA(n_components)
    pass


if __name__ == "__main__":
    main()
