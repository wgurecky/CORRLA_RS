About
================

CORRLA-RS is a package for correlation, regression and random linear algebra methods in Rust.

Methods
---

Random Linear Algebra Methods

- Randomized SVD (RSVD)

Applied Math Methods

- Active Subspace Identification by KdTree and RSVD
- Principle component analysis (PCA) by RSVD
- Proper orthogonal decomposition (POD) by RSVD
- Dynamic mode decomposition (DMD) by RSVD

Global Sensitivity Methods

- Active subspace based global sensitivity measure [1]

Correlation Analysis Methods

- Pearson correlation
- R-squared analysis

Interpolation and Response Surfaces

- N-D Radial basis function interpolation
- Gaussian process regression (TODO)

Space Sampling Methods

- Dirichlet multivariate constrained sampler
- Others TBD

Markov chain Monte Carlo (MCMC) samplers:

- Differential evolution MCMC (DEMC), parallel implementation.
- Differential evolution adaptive metropolis MCMC (DREAM), parallel implementation.

Distribution Functions

- Multivariate
  - Bivariate Copula (TODO)
    - Gaussian
    - Frank
    - Clayton
    - Gumbel
  - Pair-Copula constructions (Vine-Copula) (TODO)
- Univariate
  - Normal
  - Beta
  - Exponential
  - Kernel density

Package Build
-------------

Build this rust package with:

    cargo build

To run tests

    cargo test

for an optimized build

    cargo build --release
    cargo test --release

(Optional) Build Python Bindings
--------------------------------

The pyo3 (https://pyo3.rs) package provides some tools to simplify creating python bindings to
rust programs.

Python bindings for this rust package are provided in `src/lib_math_utils_py.rs`.

First install maturin in your python env:

    pip install maturin

Then

    cargo build --release
    maturin develop --release

Then in a python terminal to test:

    >>> import corrla_rs
    >>> import numpy as np
    >>> a = np.random.randn(100, 100)
    >>> n_rank, n_iters, n_oversamples = 10, 12, 8
    >>> corrla_rs.rsvd(a, n_rank, n_iters, n_oversamples)
    >>> x = np.random.randn(1000, 10)
    >>> y = np.random.randn(1000, 1)
    >>> est_order = 1
    >>> n_neighbors = 30
    >>> n_comps = 8
    >>> comps, vals, sensi = corrla_rs.active_ss(x, y, est_order, n_neighbors, n_comps)
    >>> print(comps, vals, sensi)


License
-------

Copyright© 2024-present, UT-Battelle, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Rust Setup
----------

Download rustup: https://www.rust-lang.org/tools/install

Get rust dev stuff:

    rustup toolchain install stable

rustup handles toolchain stuff.  Install it first.
then for lsp (rust-analyzer):

    rustup component add rust-analyzer

To update rust toolchain

    rustup update

Installing rust programs with:

    cargo install <program_name>

Third party rust libraries (tpls) must be added as dependencies in a project Cargo.toml file.
The `*` is to grab the latest version of the tpl. Ex:

    [dependencies]
    ndarray = "*"
    nalgebra = "*"

Now when doing `cargo build`, the dependencies will automagically download and build.


To speed up compilation:

The `mold` (https://github.com/rui314/mold) linker can speed up incremental builds significantly.  First, install mold (on arch: `sudo pacman -S mold`, on ubuntu: `sudo apt get install mold`) then simply use mold when doing a cargo build or run:

    mold -run cargo build

Or, to use mold by default, add to `~/.cargo/config.toml`:

    [target.x86_64-unknown-linux-gnu]
    linker = "clang"
    rustflags = ["-C", "link-arg=--ld-path=/usr/bin/mold"]

Where `/usr/bin/mold` may be different depending on the system.

Compilation times can also be somewhat reducing by using sccache (sccache package). This will maintain a local cache of compiler artifacts.

    cargo install sccache

add the following configuration to ~/.cargo/config:

    [build]
    rustc-wrapper = "sccache"

References
---

[1] Constantine, Paul G., and Paul Diaz. "Global sensitivity metrics from active subspaces." Reliability Engineering & System Safety 162 (2017): 1-13.
