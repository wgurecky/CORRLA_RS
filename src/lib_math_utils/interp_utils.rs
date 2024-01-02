/// Interpolation utility methods
use num_traits::Float;
use faer::{prelude::*, IntoNdarray};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use rand::{prelude::*};
use rand_distr::{StandardNormal, Uniform};
use rayon::prelude::*;
use ndarray::prelude::*;

// internal imports
use crate::lib_math_utils::mat_utils::*;
use crate::lib_math_utils::stats_corr::*;


// Stores persistent data for RBF interpolation
pub struct RbfInterp {
    // dimension of input space
    rbf_dim: usize,
    // locations of known support points.
    // Used to compute distance between query points and known points.
    x_known: Option<Mat<f64>>,
    // degree of augmenting polynomial
    poly_degree: usize,
    // kernel type
    kernel: Box<dyn RbfEval>,
    // interpolating coeffs
    coeffs: Option<Mat<f64>>,
}

/// Define required interface for RBF kernel
pub trait RbfEval {
    fn rbf_eval(&self, r_dist: f64) -> f64;
}

/// Linear rbf kernel
pub struct RbfKernelLin {}
impl RbfKernelLin {
    pub fn new() -> Self {
        RbfKernelLin {}
    }
}
impl RbfEval for RbfKernelLin {
    fn rbf_eval(&self, r_dist: f64) -> f64 {
        r_dist
    }
}

/// Cubic rbf kernel
pub struct RbfKernelCubic {}
impl RbfKernelCubic {
    pub fn new() -> Self {
        RbfKernelCubic {}
    }
}
impl RbfEval for RbfKernelCubic {
    fn rbf_eval(&self, r_dist: f64) -> f64 {
        r_dist * r_dist * r_dist
    }
}

/// Multi quadratic rbf kernel
pub struct RbfKernelMultiQuad {eps: f64,}
impl RbfKernelMultiQuad {
    pub fn new(eps_in: f64) -> Self {
        RbfKernelMultiQuad { eps: eps_in }
    }
}
impl RbfEval for RbfKernelMultiQuad {
    fn rbf_eval(&self, r_dist: f64) -> f64 {
        (1.0 + (self.eps * r_dist).powf(2.0)).sqrt()
    }
}

/// Gaussian rbf kernel
pub struct RbfKernelGauss {eps: f64,}
impl RbfKernelGauss {
    pub fn new(eps_in: f64) -> Self {
        RbfKernelGauss { eps: eps_in }
    }
}
impl RbfEval for RbfKernelGauss {
    fn rbf_eval(&self, r_dist: f64) -> f64 {
        (-1.0*(r_dist*self.eps).powf(2.0)).exp()
    }
}

impl RbfInterp {
    pub fn new(kernel: Box<dyn RbfEval>, dim: usize, poly_degree: usize)
        -> Self
        {
            RbfInterp {
                rbf_dim: dim,
                x_known: None,
                poly_degree: poly_degree,
                kernel: kernel,
                coeffs: None,
            }
        }

    // builds rbf kernel matrix
    fn _buildK(&self, x_in: MatRef<f64>) -> Mat<f64> {
        let mut o_mat = faer::Mat::zeros(x_in.nrows(), self.x_known.as_ref().unwrap().nrows());
        for (i, row_a) in x_in.row_chunks(1).enumerate() {
            for (j, row_b) in self.x_known.as_ref().unwrap().row_chunks(1).enumerate() {
                let r_dist = (row_a - row_b).norm_l2();
                let rbf_phi = self.kernel.rbf_eval(r_dist);
                o_mat.write(i, j, rbf_phi);
            }
        }
        o_mat
    }

    // builds vandermonde matrix
    fn _buildP(&self, x_in: MatRef<f64>) -> Mat<f64> {
        build_full_vandermonde(x_in.as_ref(), self.poly_degree)
    }

    fn _buildKP(&self, x_in: MatRef<f64>, full: bool) -> Mat<f64> {
        let matK = self._buildK(x_in);
        let matP = self._buildP(x_in);
        let mat_upper = mat_hstack(matK.as_ref(), matP.as_ref());

        // build vandermonde matrix for augmenting polynomial
        let matB = faer::Mat::zeros(matP.transpose().nrows(), matP.ncols());
        let mat_lower = mat_hstack(matP.transpose(), matB.as_ref());

        if full {
            // stack upper and lower blocks together
            mat_vstack(mat_upper.as_ref(), mat_lower.as_ref())
        }
        else {
            mat_upper
        }
    }

    pub fn fit(&mut self, x_in: MatRef<f64>, y_in: MatRef<f64>)
    {
        assert_eq!(self.rbf_dim, x_in.ncols());
        self.x_known = Some(x_in.to_owned());
        // build block matrix
        let matKP = self._buildKP(x_in, true);
        // solve for interpolating coeffs
        let matKP_inv = mat_pinv(matKP.as_ref());
        let tot_size = matKP_inv.ncols();
        let y_pad = faer::Mat::zeros(tot_size-y_in.nrows(), 1);

        let coeffs = matKP_inv * mat_vstack(y_in.as_ref(), y_pad.as_ref());
        self.coeffs = Some(coeffs);
    }

    pub fn predict(&self, x_query: MatRef<f64>)
        -> Mat<f64>
    {
        assert_eq!(self.rbf_dim, x_query.ncols());
        let test_matKP = self._buildKP(x_query, false);
        let o_mat = test_matKP * self.coeffs.as_ref().unwrap();
        o_mat
    }
}


#[cfg(test)]
mod interp_utils_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_rbf_interp() {
        let tst_cov = faer::mat![
            [1.0, 0.0],
            [0.0, 1.0],
        ];
        let x_tst = sample_mv_normal(tst_cov.as_ref(), 40);
        let tst_fn = |x1: f64, x2: f64| -> f64 {x1.sin() + x2.sin()};
        let mut y_tst = faer::Mat::zeros(x_tst.nrows(), 1);
        // evaluate the tst funciton at all samples
        for (i, sample) in x_tst.row_chunks(1).enumerate() {
            let ys: f64 = tst_fn(sample[(0, 0)], sample[(0, 1)]);
            y_tst.write(i, 0, ys);
        }
        // build the rbf interpolant
        let rbf_kern = RbfKernelMultiQuad::new(1.0);
        let mut rbf_interp_f = RbfInterp::new(Box::new(rbf_kern), 2, 1);
        rbf_interp_f.fit(x_tst.as_ref(), y_tst.as_ref());
        let x_eval = sample_mv_normal(tst_cov.as_ref(), 10);
        let y_eval = rbf_interp_f.predict(x_eval.as_ref());
        print!("y_eval: {:?}", y_eval);
    }
}
