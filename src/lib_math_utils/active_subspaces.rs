// Impl active subspace indentification methods from
//  - pre-computed samples (monte-caro or from prior LHS sampling)
//  - space filling sampling methods (moris)
//  - space filling adaptive sampling methods (moris + updated AS)
//
//
use std::cmp;
use faer::{prelude::*, IntoFaer};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
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
pub struct GradientEstimator {
    kd_tree: KdTree<f64, usize, Vec<f64>>,
    est_order: usize,
    n_nbrs: usize,
    x_mat: Mat<f64>,
    y: Mat<f64>,
    /// number of features
    k: usize,
}

/// Stores owned data for active subspace methods
pub struct ActiveSsRsvd {
    grad_est: GradientEstimator,
    pub components_: Option<Mat<f64>>,
    pub singular_vals_: Option<Mat<f64>>,
}


impl GradientEstimator {
    pub fn new(x_mat: Mat<f64>, y: Mat<f64>, est_order: usize, n_nbrs: usize)
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
            x_mat: x_mat,
            y: y,
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

    /// Estimate gradients [dy/dx0, .. dy/dxN]
    pub fn grad_at(&self, x0: Vec<f64>) -> Mat<f64>
    {
        match self.est_order {
            1 => self.est_grad_lin(x0),
            _ => self.est_grad_quad(x0)
        }
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


impl ActiveSsRsvd {
    /// x_mat is a point cloud with points <x0, x1>
    /// where y is the response.
    /// First we must compute the gradients:
    /// \grad x \cdot \grad x
    pub fn new(x_mat_in: MatRef<f64>, y_in: MatRef<f64>, order: usize, n_nbrs: usize)
        -> Self
    {
        // let means = mat_mean(x_mat_in, 1);
        // let n_samples = x_mat.nrows();
        // NOTE: creates owned copies of x and y data
        let x_mat: Mat<f64> = x_mat_in.to_owned();
        let y: Mat<f64> = y_in.to_owned();

        // build kd tree on point cloud data
        Self
        {
            grad_est: GradientEstimator::new(x_mat, y, order, n_nbrs),
            components_: None,
            singular_vals_: None,
        }
    }

    /// Compute gradients and active subspace using sample
    /// locations from x_mat rows
    pub fn fit(&mut self, x_mat: MatRef<f64>)
    {
        let k_features = x_mat.ncols();
//         let grad_outer_prod = (0..x_mat.nrows()).into_par_iter()
//             .map(|i| {
//                 let x_vec = mat_row_to_vec(x_mat.as_ref(), i);
//                 let dy_dx = self.grad_est.est_grad(x_vec);
//                 let tmp_outer_prod = &dy_dx.transpose() * &dy_dx;
//                 // grad_outer_prod = grad_outer_prod + tmp_outer_prod;
//                 tmp_outer_prod
//             }
//             ).reduce(|| faer::Mat::zeros(2,2), |x, y| x + y);
        let mut grad_outer_prod: Mat<f64> = faer::Mat::zeros(k_features, k_features);
        for i in 0..x_mat.nrows() {
            let x_vec = mat_row_to_vec(x_mat.as_ref(), i);
            let dy_dx = self.grad_est.grad_at(x_vec);
            let tmp_outer_prod = &dy_dx.transpose() * &dy_dx;
            // let tmp_outer_prod = &dy_dx * &dy_dx.transpose();
            grad_outer_prod = grad_outer_prod + tmp_outer_prod;
        }
        // compute average outer gradient product
        let avg_grad_outer_prod = grad_outer_prod * faer::scale(1. / (x_mat.nrows() as f64));

        // compute svd of outer gradient product
        let gp_svd = avg_grad_outer_prod.svd();
        self.components_ = Some(gp_svd.u().to_owned());
        self.singular_vals_ = Some(gp_svd.s_diagonal().to_owned());
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
        let grad_est = GradientEstimator::new(x_tst, y_tst, 2, 14);

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
            [0.9, 0.5],
            [0.5, 0.9],
        ];
        let x_tst = sample_mv_normal(tst_cov.as_ref(), 100);
        print!("mv norm x: {:?}", x_tst);
        let mut y_tst: Mat<f64> = faer::Mat::zeros(x_tst.nrows(), 1);
        let tst_fn = |x1: f64, x2: f64| -> f64 {0.1*x1.powf(1.) + 0.5*x2.powf(2.)};
        // evaluate the tst funciton at all samles
        for (i, sample) in x_tst.row_chunks(1).enumerate() {
            let ys: f64 = tst_fn(sample[(0, 0)], sample[(0, 1)]);
            y_tst.write(i, 0, ys);
        }

        // initialze the active subspace estimator
        let mut act_ss = ActiveSsRsvd::new(x_tst.as_ref(), y_tst.as_ref(), 1, 16);

        // use same sample locations as support points to estimate the active
        // subspace
        act_ss.fit(x_tst.as_ref());
        // print the active subspace components
        print!("\n Active subspace component directions:\n {:?}\n", act_ss.components_);
        // print the active subspace singular values
        print!("\n Active subspace singular vals:\n {:?}\n", act_ss.singular_vals_);
    }
}
