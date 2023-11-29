About
================

Corrolation and Random Linear Algebra in Rust

Methods
---

Random Linear Algebra Methods

- Randomized SVD (RSVD)

Applied Math Methods

- Active Subspace Identification by KdTree and RSVD
- PCA by RSVD
- DMD by RSVD
- POD by RSVD

Corrolation Analysis Methods

- Pearson Correlation calc
- R-squared Sensitivity analysis

Sampling Methods

- TBD

Package Build
-------------

Build this rust package with:

    cargo build

Clean proj build with

    cargo clean

To run tests

    cargo test

To run examples

    cargo run --example ex1

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
    >>> corrla_rs.rsvd(a)


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


