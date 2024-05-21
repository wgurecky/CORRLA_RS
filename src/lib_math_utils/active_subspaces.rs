/// Impl active subspace indentification methods from
///  - pre-computed samples (monte-caro or from prior LHS sampling)
///  - space filling sampling methods (moris)
///  - space filling adaptive sampling methods (moris + updated AS)
/// Ref: P. Constantine. et. al. Active subspace methods in theory and
/// practice. https://arxiv.org/pdf/1304.2070.pdf
///
use std::cmp;
use assert_approx_eq::assert_approx_eq;
use faer::{prelude::*, IntoFaer};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use faer_core::{ComplexField, RealField, c32, c64};
use faer::solvers::{Eigendecomposition};
// use kiddo::{KdTree, SquaredEuclidean};
use kdtree::KdTree;
use kdtree::ErrorKind;
use kdtree::distance::squared_euclidean;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::lib_math_utils::mat_utils::*;
use crate::lib_math_utils::random_svd::*;
use crate::lib_math_utils::stats_corr::*;
use crate::lib_math_utils::pca_rsvd::{ApplyTransform};

/// Stores owned data for gradient estimation
/// over point cloud datasets
pub struct PolyGradientEstimator {
    kd_tree: KdTree<f64, usize, Vec<f64>>,
    est_order: usize,
    n_nbrs: usize,
    x_mat: Mat<f64>,
    y: Mat<f64>,
    /// number of features
    k: usize,
}

/// Define reqired interface for gradient estimator
pub trait GradEst {
    fn grad_at(&self, x0: Vec<f64>) -> Mat<f64>;
}

/// Stores owned data for active subspace methods
pub struct ActiveSsRsvd {
    grad_est: Box<dyn GradEst>,
    pub components_: Option<Mat<f64>>,
    pub singular_vals_: Option<Mat<f64>>,
    n_comps: usize,
}

/// Interface implementation for PolyGradientEstimator
impl GradEst for PolyGradientEstimator {
    /// Estimate gradients [dy/dx_i, .. dy/dx_N]|_x0
    fn grad_at(&self, x0: Vec<f64>) -> Mat<f64>
    {
        match self.est_order {
            1 => self.est_grad_lin(x0),
            2 => self.est_grad_quad(x0),
            _ => panic!("Not implemented est order: {:?}", self.est_order)
        }
    }
}

/// Polynomial gradient estimator impl
impl PolyGradientEstimator {
    pub fn new(x_mat: MatRef<f64>, y: MatRef<f64>, est_order: usize, n_nbrs: usize)
    -> Self
    {
        let k = x_mat.ncols();
        let mut kd_tree = KdTree::new(k);
        // fill kd tree with points
        for i in 0..x_mat.nrows() {
            // convert row to vec
            let tmp_vec: Vec<f64> = mat_row_to_vec(x_mat.as_ref(), i);
            // add vec to kdtree
            kd_tree.add(tmp_vec, i).unwrap();
        }

        Self {
            kd_tree: kd_tree,
            est_order: est_order,
            n_nbrs: n_nbrs,
            x_mat: x_mat.to_owned(),
            y: y.to_owned(),
            k: k
        }
    }

    /// Find nearest n_nbrs points to x0
    fn nearest_points(&self, x0: &Vec<f64>, n_nbrs: usize) -> (Mat<f64>, Mat<f64>)
    {
        // get the nearest n_nbrs neighboring points to x0
        let nearest: Vec<_> = self.kd_tree.nearest(
            x0, n_nbrs, &squared_euclidean).unwrap();
        // allocate storage for nearest points
        assert!(self.k == x0.len());
        let mut x_nbr = Array2::<f64>::zeros((nearest.len(), self.k));
        let mut y_nbr = Array2::<f64>::zeros((nearest.len(), 1));
        // get the corrosponding x, y value pairs
        for i in 0..nearest.len() {
            // .1 stores index, .0 stores distance to node
            let n = nearest[i].1.to_owned();
            let n_x = mat_row_to_vec(self.x_mat.as_ref(), n);
            let n_y = mat_row_to_vec(self.y.as_ref(), n);
            x_nbr.row_mut(i).assign(&Array1::from_vec(n_x));
            y_nbr.row_mut(i).assign(&Array1::from_vec(n_y));
        }
        // convert ndarrays into faer matrix
        let x_nbr_faer = (&x_nbr).view().into_faer();
        let y_nbr_faer = (&y_nbr).view().into_faer();
        (x_nbr_faer.to_owned(), y_nbr_faer.to_owned())
    }

    /// Estimate gradients [dy/dx0, .. dy/dxN] at x0 using linear fit
    pub fn est_grad_lin(&self, x0: Vec<f64>) -> Mat<f64>
    {
        // ensure enough samples avail to fit hyper-plane
        assert!(self.x_mat.nrows() > self.k + 1);
        assert!(self.n_nbrs > self.k + 1);
        let (x_nbr_faer, y_nbr_faer) = self.nearest_points(&x0, self.n_nbrs);
        let grads = jac_from_lin(x_nbr_faer.as_ref(), y_nbr_faer.as_ref());
        grads
    }

    /// Estimate gradients [dy/dx0, .. dy/dxN] at x0 using quadratic fit
    pub fn est_grad_quad(&self, x0: Vec<f64>) -> Mat<f64>
    {
        // ensure enough samples avail to fit hyper-quadratic
        assert!(self.x_mat.nrows() > self.k*(self.k+3)/2);
        assert!(self.n_nbrs > self.k*(self.k+3)/2);
        let (x_nbr_faer, y_nbr_faer) = self.nearest_points(&x0, self.n_nbrs);
        // construct polynomial approx over neighbors
        let poly_coeffs = quad_fit(x_nbr_faer.as_ref(), y_nbr_faer.as_ref());
        // calc gradient of poly approx at x0
        let mut tmp_x0_arr = Array2::zeros((1, x0.len()));
        tmp_x0_arr.row_mut(0).assign(&Array1::from_vec(x0));
        let x0_faer = tmp_x0_arr.view().into_faer();
        let grads = jac_from_quad(x0_faer.as_ref(), poly_coeffs.as_ref());
        grads
    }
}

/// Active subspace estimator impl
impl ActiveSsRsvd {
    /// Init the active subspace estimator
    pub fn new<T>(grad_est_in: T, n_comps: usize)
        -> Self
        where
        T: GradEst + 'static
    {
        Self
        {
            grad_est: Box::new(grad_est_in),
            components_: None,
            singular_vals_: None,
            n_comps: n_comps,
        }
    }

    fn create_grad_mat(&self, x_mat: MatRef<f64>) -> Mat<f64>
    {
        let k_features = x_mat.ncols();
        let mut grad_mat: Mat<f64> = faer::Mat::zeros(k_features, x_mat.nrows());
        // Ref: P. Constantine. et. al. Active subspace methods in theory and
        // practice. https://arxiv.org/pdf/1304.2070.pdf
        // Eqs. 2.16 - 2.18
        for i in 0..x_mat.nrows() {
            let x_vec = mat_row_to_vec(x_mat.as_ref(), i);
            let dy_dx = self.grad_est.grad_at(x_vec);
            grad_mat.col_as_slice_mut(i).iter_mut().enumerate().for_each(
                |(j, ele)| *ele = dy_dx.read(0, j));
        }
        grad_mat
    }

    /// Compute gradients and active subspace using sample
    /// locations from x_mat rows
    pub fn fit_svd(&mut self, x_mat: MatRef<f64>, n_iter: Option<usize>, n_oversamples: Option<usize>)
    {
        let k_features = x_mat.ncols();
        let grad_mat = self.create_grad_mat(x_mat);
        let grad_mat_sc = grad_mat * faer::scale(1. / (x_mat.nrows() as f64).sqrt());

        // compute svd of the gradient matrix
        let (ur, sr, _vr) = random_svd(grad_mat_sc.as_ref(),
            cmp::min(k_features, self.n_comps),
            n_iter.unwrap_or(8), n_oversamples.unwrap_or(10));
        self.components_ = Some(ur);
        self.singular_vals_ = Some(mat_colvec_to_diag(sr.as_ref()));
    }

    pub fn fit(&mut self, x_mat: MatRef<f64>)
    {
        let grad_mat = self.create_grad_mat(x_mat);
        let grad_mat_sc = grad_mat.as_ref() * grad_mat.transpose() * faer::scale(1. / (x_mat.nrows() as f64));
        assert!(grad_mat_sc.nrows() == x_mat.ncols());
        assert!(grad_mat_sc.ncols() == x_mat.ncols());
        // all eigenvalues of a real sym mat are real
        let evd: Eigendecomposition<c64> = grad_mat_sc.eigendecomposition();
        let eigs = evd.s_diagonal();
        let eig_vs = evd.u();
        // split real from complex part, discard imag parts (zero)
        let (real_eigs, _imag_eigs) =
            mat_parts_from_complex(mat_colmat_to_diag(eigs).as_ref());
        let (real_eig_vs, _imag_eig_vs) =
            mat_parts_from_complex(eig_vs);
        // sort eigenvalues and eigenvectors from largest to smallest
        let (sorted_singular_vals, sorted_components) =
            sort_evd(real_eigs.as_ref(), real_eig_vs.as_ref());
        self.components_ = Some(sorted_components);
        self.singular_vals_ = Some(sorted_singular_vals);
    }

    /// Compute variable sensitivity measure based on:
    /// Constantine, Paul G., and Paul Diaz.
    /// "Global sensitivity metrics from active subspaces."
    /// Reliability Engineering & System Safety 162 (2017): 1-13.
    /// https://arxiv.org/pdf/1510.04361
    /// Eq 22 based on eigenvalue decomp of the gradient matrix
    pub fn var_diag_evd_sensi(&self) -> Vec<f64> {
        let ndim = self.singular_vals_.as_ref().unwrap().nrows();
        let mut evd_sensi: Vec<f64> = Vec::new();
        let grad_mat_reconstructed = self.components_.as_ref().unwrap().transpose() *
            self.singular_vals_.as_ref().unwrap().as_ref() *
            self.components_.as_ref().unwrap().as_ref();
        for i in 0..ndim {
            evd_sensi.push(grad_mat_reconstructed.read(i, i));
        }
        evd_sensi
    }

    /// Project a data matrix of higher dimension onto the active subspace components
    pub fn transform(&self, x_mat: MatRef<f64>) -> Mat<f64>
    {
        // components are mxr, with m being k_features, and r is subspace dim
        // x_mat is nxm with n==number of samples, m is k_features
        // output is x_mat * components with nxr
        x_mat * self.components()
    }

    /// Project a reduced data matrix back to original space
    pub fn inv_transform(&self, x_mat: MatRef<f64>) -> Mat<f64>
    {
        // x_mat is nxr with n==number of samples, r is subspace components
        assert!(x_mat.ncols() == self.n_comps);
        x_mat * self.components().transpose()
    }

    /// Getter for active subspace components
    pub fn components(&self) -> MatRef<f64> {
        self.components_.as_ref().unwrap().get(.., 0..self.n_comps)
    }

    /// Getter for singular_vals
    pub fn singular_vals(&self) -> MatRef<f64> {
        self.singular_vals_.as_ref().unwrap().get(.., 0..self.n_comps)
    }
}


#[cfg(test)]
mod active_subspace_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_grad_est() {
        let tst_cov = faer::mat![
            [0.9, 0.5],
            [0.5, 0.9],
        ];
        let x_tst = sample_mv_normal(tst_cov.as_ref(), 100);
        print!("mv norm x: {:?}", x_tst);
        let mut y_tst: Mat<f64> = faer::Mat::zeros(x_tst.nrows(), 1);
        let tst_fn = |x1: f64, x2: f64| -> f64 {x1.powf(2.) + x2.powf(2.)};
        // evaluate the tst funciton at all samles
        for (i, sample) in x_tst.row_chunks(1).enumerate() {
            let ys: f64 = tst_fn(sample[(0, 0)], sample[(0, 1)]);
            y_tst.write(i, 0, ys);
        }

        // create gradient estimator
        let grad_est = PolyGradientEstimator::new(
            x_tst.as_ref(), y_tst.as_ref(), 2, 14);

        // eval gradient at one location
        let eval_x_0: Vec<f64> = vec![0., 0.];
        let grad_y_0 = grad_est.grad_at(eval_x_0);
        print!("at x=[0., 0.], est grad= {:?} \n", grad_y_0);
        // quadratic flat at x=0,0
        mat_mat_approx_eq(grad_y_0.as_ref(), mat![[0., 0.]].as_ref(), 1.0e-2f64);

        let eval_x_1: Vec<f64> = vec![1., 0.];
        let grad_y_1 = grad_est.grad_at(eval_x_1);
        let eval_x_2: Vec<f64> = vec![-1., 0.];
        let grad_y_2 = grad_est.grad_at(eval_x_2);
        print!("at x=[1., 0.], est grad= {:?} \n", grad_y_1);
        print!("at x=[-1., 0.], est grad= {:?} \n", grad_y_2);
        mat_mat_approx_eq(grad_y_1.as_ref(), mat![[2.0, 0.]].as_ref(), 1.0e-2f64);
        mat_mat_approx_eq(grad_y_1.as_ref(), (faer::scale(-1.)*grad_y_2).as_ref(), 1.0e-2f64);
    }

    #[test]
    fn test_active_ss() {
        let tst_cov = faer::mat![
            [0.9, 0.5, 0.5],
            [0.5, 0.9, 0.5],
            [0.5, 0.5, 0.9],
        ];
        let x_tst = sample_mv_normal(tst_cov.as_ref(), 100);
        print!("mv norm x: {:?}", x_tst);
        let mut y_tst: Mat<f64> = faer::Mat::zeros(x_tst.nrows(), 1);
        let tst_fn = |x1: f64, x2: f64, x3: f64| -> f64 {
            0.2*x1.powf(1.) + 0.5*x2.powf(2.) + 0.10*x3*x1};
        // evaluate the tst funciton at all samles
        for (i, sample) in x_tst.row_chunks(1).enumerate() {
            let ys: f64 = tst_fn(sample[(0, 0)], sample[(0, 1)], sample[(0, 2)]);
            y_tst.write(i, 0, ys);
        }

        // initialze the gradient estimator
        let grad_est = PolyGradientEstimator::new(
            x_tst.as_ref(), y_tst.as_ref(), 2, 14);

        // initialze the active subspace estimator, supplying a
        // valid gradient estimator as input to init
        let n_comps = 2;
        let mut act_ss = ActiveSsRsvd::new(grad_est, n_comps);

        // use same sample locations as support points to estimate the active
        // subspace
        act_ss.fit(x_tst.as_ref());
        // print the active subspace components
        print!("\n Active subspace component directions:\n {:?}\n", act_ss.components_);
        // print the active subspace singular values
        print!("\n Active subspace singular vals:\n {:?}\n", act_ss.singular_vals_);
        // Known higher gradient variability along x2 so first active component should be
        // dominated by the x2 direction.
        assert!(act_ss.components().read(0, 0).abs() < act_ss.components().read(1, 0).abs());
        assert!(act_ss.singular_vals().read(0, 0) > act_ss.singular_vals().read(1, 1));

        // check the grad estimator
        let eval_x_0: Vec<f64> = vec![0., 1., 0.];
        let grad_y_0 = act_ss.grad_est.grad_at(eval_x_0);
        print!("at x=[0., 1., 0.], est grad= {:?} \n", grad_y_0);
        mat_mat_approx_eq(grad_y_0.as_ref(), mat![[0.2, 1., 0.]].as_ref(), 1.0e-1f64);

        // project the original sample locations into 2D active subspace
        let tr_x_tst = act_ss.transform(x_tst.as_ref());
        assert!(tr_x_tst.ncols() == n_comps);
        let tr_inv_x_tst = act_ss.inv_transform(tr_x_tst.as_ref());
        assert!(tr_inv_x_tst.ncols() == 3);
        // ensure the points mapped back to their original positions
        // assert_approx_eq!(tr_inv_x_tst.read(0, 0), x_tst.read(0, 0));
        // assert_approx_eq!(tr_inv_x_tst.read(0, 1), x_tst.read(0, 1));

        // test the sensitivity methods
        let sens_coeffs = act_ss.var_diag_evd_sensi();
        println!("\n Active SS sensitivity coeffs: {:?} \n", sens_coeffs);
        assert!(sens_coeffs.len() == 3);
        // x2 direction should dominate sensitivity metrics
        assert!(sens_coeffs[1] > sens_coeffs[0]);
        assert!(sens_coeffs[1] > sens_coeffs[2]);
    }
}
