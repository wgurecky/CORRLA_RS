/// Dynamic Mode Decomposition impl.
///
/// Ref: Proctor, J., Brunton, S., Kutz, J.
/// Dynamic Mode Decomposition with Control
///
/// Represents dynamic systems using a linear model
/// x_t+1 = A x_t + B u_t
///
/// Where u_t is the control or forcing term
/// and x_t is the state vector.
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut, ColMut};
use num_traits::Float;

// internal imports
use crate::lib_math_utils::mat_utils::*;
use crate::lib_math_utils::random_svd::*;
use crate::lib_math_utils::interp_utils::*;

pub struct DMDc {
    // number of supplied data snapshots
    n_snapshots: usize,
    // number of state vars
    n_x: usize,
    // number of control inputs
    n_u: usize,
    // number of DMD modes to compute and store for reconstructing output
    n_modes: usize,
    // times at which snapshots were collected
    t_abscissa: Mat<f64>,
    // input space,
    omega: Mat<f64>,
    // DMD mode weight storage (similar to eig vals)
    lambdas: Option<Mat<f64>>,
    // DMD mode storage (similar to eigenvecs)
    modes: Option<Mat<f64>>,
    // DMD mode weight storage (similar to eig vals)
    _basis: Option<Mat<f64>>,
    // DMD mode storage (similar to eigenvecs)
    _B: Option<Mat<f64>>,
    _A: Option<Mat<f64>>,
}


impl DMDc {
    pub fn new(x_data: MatRef<f64>, u_data: MatRef<f64>, t: MatRef<f64>, n_modes: usize) -> Self {
        let mut dmdc_inst = Self {
            n_snapshots: x_data.ncols(),
            n_x: x_data.nrows(),
            n_u: u_data.nrows(),
            n_modes: n_modes,
            t_abscissa: t.to_owned(),
            omega: mat_vstack(x_data, u_data),
            lambdas: None,
            modes: None,
            _basis: None,
            _B: None,
            _A: None,
        };
        dmdc_inst._calc_dmdc_modes();
        dmdc_inst
    }

    /// Computes DMD modes
    fn _calc_dmdc_modes(&mut self) {
        // compute SVD of input space
        let (u_til, s_til, v_til) = random_svd(self.omega.as_ref(), self.n_modes, 10, 10);

        let u_til_1 = u_til.as_ref().submatrix(
            0, 0, self.n_x, u_til.ncols());
        let u_til_2 = u_til.as_ref().submatrix(
            self.n_x, 0,
            u_til.nrows()-self.n_x, u_til.ncols());

        // compute SVD of output space
        let (u_hat, _s_hat, _v_hat) = random_svd(self._Y(), self.n_modes, 10, 10);

        // from eq 29 in Proctor. et. al DMDc
        let a_til =
            u_hat.transpose()
            * (self._Y()
            * (v_til.as_ref()
            * (s_til.qr().inverse()
            * (u_til_1.transpose() * u_hat.as_ref()))));

        // from eq 30 in Proctor. et. al DMDc
        let b_til =
            u_hat.transpose()
            * (self._Y()
            * (v_til.as_ref()
            * (s_til.qr().inverse()
            * (u_til_2.transpose()))));
        self._basis = Some(u_hat.clone());
        self._A = Some(a_til);
        self._B = Some(u_hat.as_ref()*b_til);

        self._calc_modes(v_til.as_ref(), s_til.as_ref(), u_til_1.as_ref(), u_hat.as_ref())
    }

    /// Computes eigenvalues and eigenvectors of a_tilde
    fn _calc_eigs(&self) -> (Mat<f64>, Mat<f64>) {
        let ev = (self._A.as_ref()).unwrap().eigendecomposition::<f64>();
        let a_til_eigenvectors = ev.u();
        let a_til_eigenvalues = ev.s_diagonal().as_2d();
        (a_til_eigenvalues.to_owned(), a_til_eigenvectors.to_owned())
    }

    /// Computes DMD modes
    fn _calc_modes(&mut self, v_til: MatRef<f64>, s_til: MatRef<f64>, u_til_1: MatRef<f64>, u_hat: MatRef<f64>) {
        let (lambdas, w) = self._calc_eigs();
        self.lambdas = Some(lambdas);
        // from eq 36 in Proctor. et. al DMDc
        self.modes = Some(self._Y() * v_til * s_til.qr().inverse() * u_til_1 * u_hat * w);
    }

    /// Return the snapshots of x_data from 0, N-1
    fn _X(&self) -> MatRef<f64> {
        let out_x = self.omega.as_ref().submatrix(
            0, 0, self.omega.nrows()-self.n_u, self.omega.ncols()-1);
        out_x
    }

    /// Return the snapshots of x_data from 1, N
    fn _Y(&self) -> MatRef<f64> {
        // Construct oputput space, without forcing inputs
        // [x_1, x_2, ... x_N]
        let out_x = self.omega.as_ref().submatrix(
            0, 1, self.omega.nrows()-self.n_u, self.omega.ncols()-1);
        out_x
    }

    /// Step the physics forward using DMDc
    /// x_0 should be a single column matrix
    /// u_input should be a single column matrix of control vars
    fn predict(&self, x_0: MatRef<f64>, u_input: MatRef<f64>) -> Mat<f64> {
        assert_eq!(x_0.nrows(), self.n_x);
        assert_eq!(x_0.ncols(), 1);
        assert_eq!(u_input.nrows(), self.n_u);
        assert_eq!(u_input.ncols(), 1);
        // reconstrut A from eigendecomposition
        let a_til = self.modes.as_ref().unwrap() *
                    self.lambdas.as_ref().unwrap() *
                    mat_pinv(self.modes.as_ref().unwrap().as_ref());
        let next_x = a_til * x_0 + self._B.as_ref().unwrap() * u_input;
        next_x
    }

    /// Predict multiple steps
    /// x_0 should be a single column matrix (n_x, 1)
    /// u_seq should have hsape (n_u, n_times)
    fn predict_multiple(&self, x_0: MatRef<f64>, u_seq: MatRef<f64>) -> Mat<f64> {
        assert_eq!(x_0.nrows(), self.n_x);
        assert_eq!(x_0.ncols(), 1);
        assert_eq!(u_seq.nrows(), self.n_u);
        // reconstrut A from eigendecomposition
        let a_til = self.modes.as_ref().unwrap() *
                    self.lambdas.as_ref().unwrap() *
                    mat_pinv(self.modes.as_ref().unwrap().as_ref());

        // storage
        let mut x_cur = x_0;
        let mut next_x: Mat<f64> = x_0.to_owned();
        let mut x_out = faer::Mat::zeros(self.n_x, u_seq.ncols());

        // step the dynamics forward, for each u in u_input
        let mut j = 0;
        let col_iter = u_seq.col_chunks(1);
        for u_input in col_iter {
            next_x = a_til.as_ref() * x_cur + self._B.as_ref().unwrap() * u_input;
            for i in 0..self.n_x {
                x_out.write(i, j, next_x.read(i, 0));
            }
            // store next_x
            x_cur = next_x.as_ref();
            j = j + 1;
        }

        x_out
    }
}
