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
use faer_core::{ComplexField, RealField, c32, c64};
use faer::solvers::{Eigendecomposition};
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
    dt_snapshots: f64,
    // input space,
    omega: Mat<f64>,
    // DMD mode weight storage (similar to eig vals)
    lambdas: Option<Mat<c64>>,
    // DMD mode storage (similar to eigenvecs)
    modes_re: Option<Mat<f64>>,
    modes_im: Option<Mat<f64>>,
    // low rank DMD operator storage
    _B: Option<Mat<f64>>,
    _A: Option<Mat<f64>>,
    _basis: Option<Mat<f64>>,
}


impl DMDc {
    pub fn new(x_data: MatRef<f64>, u_data: MatRef<f64>, dt: f64, n_modes: usize, n_iters: usize) -> Self {
        let mut dmdc_inst = Self {
            n_snapshots: x_data.ncols(),
            n_x: x_data.nrows(),
            n_u: u_data.nrows(),
            n_modes: n_modes,
            dt_snapshots: dt,
            omega: mat_vstack(x_data, u_data),
            lambdas: None,
            modes_re: None,
            modes_im: None,
            _basis: None,
            _B: None,
            _A: None,
        };
        dmdc_inst._calc_dmdc_modes(n_iters);
        dmdc_inst
    }

    /// Computes DMD modes
    fn _calc_dmdc_modes(&mut self, n_iters: usize) {
        // compute SVD of input space
        // let (u_til_rsvd, s_til_rsvd, v_til_rsvd) = random_svd(self._X(), self.n_modes, n_iters, 12);
        let (u_til, s_til, v_til_) = mat_truncated_svd(self._X(), self.n_modes);

        // let v_til = v_til_.transpose().to_owned();
        let v_til = v_til_.to_owned();

        let u_til_1 = u_til.as_ref().submatrix(
            0, 0, self.n_x, u_til.ncols());
        let u_til_2 = u_til.as_ref().submatrix(
            self.n_x, 0,
            self.n_u, u_til.ncols());

        // compute SVD of output space
        // let (u_hat_rsvd, _s_hat, _v_hat) = random_svd(self._Y(), self.n_modes, n_iters, 12);
        let (u_hat, _s_hat, _v_hat) = mat_truncated_svd(self._Y(), self.n_modes);


        let s_til_diag = mat_colvec_to_diag(s_til.as_ref());
        let s_til_inv = mat_pinv_diag(s_til_diag.as_ref());

        // from eq 29 in Proctor. et. al DMDc
        let a_til =
            u_hat.as_ref().transpose()
            * self._Y()
            * v_til.as_ref()
            * s_til_inv.as_ref()
            * u_til_1.transpose() * u_hat.as_ref();

        // from eq 30 in Proctor. et. al DMDc
        let b_til: Mat<f64> =
            u_hat.as_ref().transpose()
            * self._Y()
            * v_til.as_ref()
            * s_til_inv.as_ref()
            * u_til_2.transpose();
        //
        self._basis = Some(u_hat.clone());
        self._A = Some(a_til);
        self._B = Some(u_hat.as_ref()*b_til);

        self._calc_modes(v_til.as_ref(), s_til_diag.as_ref(), u_til_1.as_ref(), u_hat.as_ref())
    }

    /// Computes eigenvalues and eigenvectors of a_tilde
    fn _calc_eigs(&self) -> (Mat<c64>, Mat<f64>, Mat<f64>) {
        //let ev: Eigendecomposition<c64> = Eigendecomposition::new_from_real(
        //    self._A.as_ref().unwrap().as_ref());
        let ev: Eigendecomposition<c64> = (self._A.as_ref()).unwrap().eigendecomposition();
        let a_til_eigenvectors = ev.u();
        let a_til_eigenvalues = ev.s_diagonal().as_2d();
        // convert to real and imag components
        let (a_til_eigenvectors_re, a_til_eigenvectors_im) =
            mat_parts_from_complex(a_til_eigenvectors);

        (a_til_eigenvalues.to_owned(),
         a_til_eigenvectors_re.to_owned(),
         a_til_eigenvectors_im.to_owned())
    }

    /// Computes DMD modes
    fn _calc_modes(&mut self, v_til: MatRef<f64>, s_til: MatRef<f64>, u_til_1: MatRef<f64>, u_hat: MatRef<f64>) {
        let (lambdas, w_re, w_im) = self._calc_eigs();
        let lambdas_diag: Mat<c64> = mat_colvec_to_diag(lambdas.as_ref());
        self.lambdas = Some(lambdas_diag);
        // from eq 36 in Proctor. et. al DMDc
        // BUT we only need the real part of the modes, since
        // when we recombine with
        self.modes_re = Some(
            self._Y()
            * (v_til
            * (mat_pinv_diag(s_til)
            * (u_til_1.transpose()
            * (u_hat * w_re.as_ref())))));
        self.modes_im = Some(
            self._Y()
            * (v_til
            * (mat_pinv_diag(s_til)
            * (u_til_1.transpose()
            * (u_hat * w_im.as_ref())))));
    }

    /// Return the snapshots of x_data from 0, N-1
    fn _X(&self) -> MatRef<f64> {
        let out_x = self.omega.as_ref().submatrix(
            0, 0, self.omega.nrows(), self.omega.ncols()-1);
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

    /// Estimated A operator by eigendecomp
    pub fn est_a_til(&self) -> Mat<f64> {
        // build temp complex matricies from real/imag components
        let modes_comp = mat_complex_from_parts(
            self.modes_re.as_ref().unwrap().as_ref(),
            self.modes_im.as_ref().unwrap().as_ref());
        let a_til_comp = modes_comp.as_ref() *
                    self.lambdas.as_ref().unwrap() *
                    mat_pinv_comp(modes_comp.as_ref());
        let (atil_re, _atil_im) = mat_parts_from_complex(a_til_comp.as_ref());
        atil_re
    }

    /// Estimated B operator
    pub fn est_b_til(&self) -> MatRef<f64> {
        self._B.as_ref().unwrap().as_ref()
    }

    /// Step the physics forward using DMDc
    /// x_0 should be a single column matrix
    /// u_input should be a single column matrix of control vars
    pub fn predict(&self, x_0: MatRef<f64>, u_input: MatRef<f64>) -> Mat<f64> {
        assert_eq!(x_0.nrows(), self.n_x);
        assert_eq!(x_0.ncols(), 1);
        assert_eq!(u_input.nrows(), self.n_u);
        assert_eq!(u_input.ncols(), 1);
        // reconstruct A from eigendecomposition
        let a_til = self.est_a_til();
        let next_x = a_til * x_0 + self._B.as_ref().unwrap() * u_input;
        next_x
    }

    /// Predict multiple steps
    /// x_0 should be a single column matrix (n_x, 1)
    /// u_seq should have hsape (n_u, n_times)
    pub fn predict_multiple(&self, x_0: MatRef<f64>, u_seq: MatRef<f64>) -> Mat<f64> {
        assert_eq!(x_0.nrows(), self.n_x);
        assert_eq!(x_0.ncols(), 1);
        assert_eq!(u_seq.nrows(), self.n_u);
        // reconstruct A from eigendecomposition
        let a_til = self.est_a_til();

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

#[cfg(test)]
mod dmd_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_dmdc() {
        let nx = 20;
        let x_points = mat_linspace::<f64>(0.0, 10.0, nx);
        let nt = 40;
        let n_snapshots = nt.clone();
        let t_points = mat_linspace::<f64>(0.0, 10.0, nt);

        // build control input sequence
        let u_seq: Vec<_> = t_points.col_as_slice(0).into_iter().map(
            |t| { 0.2 }
            ).collect();
        let u_mat = mat_from_vec(&u_seq);
        let u_mat = u_mat.as_ref().transpose();

        // build snapshots.
        // Exponential growth and decaying oscillations
        let mut p_snapshots: Mat<f64> = faer::Mat::zeros(nx, n_snapshots);
        for n in 0..n_snapshots {
            let t = t_points[(n, 0)];
            let u = u_seq[n];
            for i in 0..x_points.nrows() {
                let x = x_points.read(i, 0);
                let p = (x+0.2*t).sin()*(u*t).exp();
                p_snapshots.write(i, n, p);
            }
        }
        // println!("p_snapshots: {:?}", p_snapshots.as_ref());
        // check data shapes
        println!("x_data shape: {:?}, {:?}", p_snapshots.nrows(), p_snapshots.ncols());
        println!("u_data shape: {:?}, {:?}", u_mat.nrows(), u_mat.ncols());

        // build DMDc model
        let dmdc_model = DMDc::new(p_snapshots.as_ref(), u_mat.as_ref(), 1.0, 3, 4);

        // test the DMDc model
        let estimated_a_op = dmdc_model.est_a_til();
        let estimated_b_op = dmdc_model.est_b_til();
        assert_eq!(estimated_a_op.ncols(), nx);
        assert_eq!(estimated_a_op.nrows(), nx);
        assert_eq!(estimated_b_op.nrows(), nx);

        // Make some predictions
        // take first snapshot as initial condition
        let x0 = p_snapshots.as_ref().submatrix(
            0, 0,
            p_snapshots.nrows(), 1
            );
        let p_predicted = dmdc_model.predict_multiple(x0.as_ref(), u_mat.as_ref());

        // get the 20th snapshot (true data)
        let p20_expected = p_snapshots.as_ref().submatrix(
            0, 1,
            p_snapshots.nrows(), 1
            );
        println!("Expected: {:?}", p20_expected);

        // get the 19th predicted state (estimated data),
        // 0th state was supplied as initial condition so offset is needed
        let p20_predicted = p_predicted.as_ref().submatrix(
            0, 0,
            p_snapshots.nrows(), 1
            );
        println!("Predicted: {:?}", p20_predicted);
        println!("DMDc Eigs: {:?}", dmdc_model.lambdas.as_ref());
    }
}
