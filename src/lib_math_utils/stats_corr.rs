/// Calculate correlation coefficients and
/// compute and manipulate covariance matricies
///
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};

// Internal imports
use crate::lib_math_utils::mat_utils::*;


/// Computes the linear correlation coefficient
/// between N random varibles.
/// Each column represents a feature dim
pub fn pearson_corr(x: MatRef<f64>) -> Mat<f64>
{
    // subtract mean from each column to center features
    // and divide by std devs
    let x_zc = zcenter_mat_col(x);
    let n_samples = x.nrows();
    // compute covariance matrix, x.T * x
    let mut pcov = faer::Mat::zeros(x.ncols(), x.ncols());
    // let pcov = (x_zc.transpose() * x_zc) * faer::scale(1. / (n_samples as f64 - 1.));
    par_matmul_helper(
        pcov.as_mut(),
        x_zc.transpose().as_ref(),
        x_zc.as_ref(), 1.0, 8);
    pcov * faer::scale(1. / (n_samples as f64 - 1.))
}

/// Computes the covariance matrix given matrix of samples
/// where each col contains the samples form a feature.
pub fn mat_cov_centered(x: MatRef<f64>) -> Mat<f64>
{
    let x_c = center_mat_col(x);
    let n_samples = x.nrows();
    // let cov = (x_c.transpose() * x_c) * faer::scale(1. / (n_samples as f64 - 1.));
    let mut cov = faer::Mat::zeros(x.ncols(), x.ncols());
    par_matmul_helper(
        cov.as_mut(),
        x_c.transpose().as_ref(),
        x_c.as_ref(), 1.0, 8);
    cov * faer::scale(1. / (n_samples as f64 - 1.))
}

/// Draws n samples from multivar gaussian
pub fn sample_mv_normal(cov: MatRef<f64>, n: usize) -> Mat<f64>
{
    let mut res = faer::Mat::zeros(n, cov.ncols());
    for (_i, mut res_row) in res.row_chunks_mut(1).enumerate() {
        // draw random normal vector
        let tmp_rv: Mat<f64> = random_mat_normal(cov.ncols(), 1);
        let sample_mv_norm = cov * tmp_rv;
        for j in 0..sample_mv_norm.nrows() {
            res_row[(0, j)] = sample_mv_norm.read(j, 0);
        }
    }
    res
}

/// Applies the sandwitch formula to matrix C
/// y_sigma = J * S * J^T
/// where C is a covariance matrix and J is a linear sensitivity
/// matrix (jacobian), typically found by finite difference.
pub fn sandwich_prop(cov: MatRef<f64>, jac: MatRef<f64>) -> Mat<f64>
{
    let sig_y = jac * cov * jac.transpose();
    sig_y
}

/// R^2 sensitivity analysis.  Uses linear correlation
/// coeffs to approximate strength of contribution of each
/// feature on the output, as measured by linear effect.
/// Ref: F. Bostelmann, et al. Extention of SCALE/Sampler sensitivity analysis.
/// Annals of Nuclear Energy. vol 165. 2022.
pub fn rsquared_sens(x: MatRef<f64>, y: MatRef<f64>, cor_dof: bool) -> Mat<f64>
{
    let n_samples = x.nrows();
    let k_features = x.ncols();
    // stack the model outputs, y, onto col of x
    let xy = mat_hstack(x, y);
    // compute pearson correlation coeffs of combined matrix
    let r_xy = pearson_corr(xy.as_ref());
    // extract corr coeffs between only input features
    let r_xx = r_xy.get(0..r_xy.nrows()-1, 0..r_xy.ncols()-1);
    // Extract last column from full corr matrix. This
    // is a vec of correlation between each input feature and
    // the output: r_y = [rho(X_i, Y), ... rho(X_N, Y)]
    let r_y = r_xy.get(0..r_xy.ncols()-1, r_xy.ncols()-1..r_xy.ncols());
    // compute R_sqr = r_y * r_xx^-1, r_y
    print!("r_xy {:?}", r_xy);
    print!("r_xx {:?}", r_xx);
    print!("r_y {:?}", r_y);
    print!("pinv r_xx {:?}", mat_pinv(r_xx));
    let mut r_sqr = r_y.transpose() * mat_pinv(r_xx) * r_y;
    print!("r_sqr {:?}", r_sqr);
    // correct for low sample size
    if cor_dof {
        r_sqr = faer::scale(-1.0) * r_sqr;
        mat_scalar_add(r_sqr.as_mut(), 1.0);
        let dof_factor = (n_samples as f64 - 1.) /
            (n_samples as f64 - k_features as f64 - 1.);
        r_sqr = r_sqr * faer::scale(dof_factor);
        r_sqr = faer::scale(-1.0) * r_sqr;
        mat_scalar_add(r_sqr.as_mut(), 1.0);
    }
    r_sqr
}


/// computes matrix with new columns defined by
/// x1*x2, ... x1*xN, x2*x3, ... x2*xN, xN*xN-1
pub fn mat_col_interactions(x: MatRef<f64>, include_self_interactions: bool) -> Mat<f64>
{
    // size the output matrix
    let mut out_ncols: usize = 0;
    // calculate total number of interactions
    for i_a in 0..x.ncols() {
        for i_b in i_a..x.ncols() {
            if i_a == i_b && !include_self_interactions {
                continue;
            }
            out_ncols += 1;
        }
    }
    let mut out_m = faer::Mat::zeros(x.nrows(), out_ncols);
    let mut cur_col: usize = 0;
    for i_a in 0..x.ncols() {
        // let c_a = x.get(.., i_a..i_a+1);
        let c_a = x.col(i_a);
        for i_b in i_a..x.ncols() {
            if i_a == i_b && !include_self_interactions {
                continue;
            }
            let c_b = x.col(i_b);
            for ri in 0..x.nrows() {
                out_m.write(ri, cur_col, c_a.read(ri) * c_b.read(ri));
            }
            cur_col += 1;
        }
    }
    out_m
}

/// Compute x1*b1 + x2*b2 + ... + b0 fit to data using simple
/// vanermonde matrix moore-penrose inversion
pub fn linear_fit(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64>
{
    // append 1 vector to x data matrix for
    // bias term to construct vandermonde matrix
    let mut ones_col = faer::Mat::zeros(x.nrows(), 1);
    ones_col.fill(1.0);
    let vand_a = mat_hstack(x, ones_col.as_ref());
    // fit the model
    let vand_a_pinv = mat_pinv(vand_a.as_ref());
    // return the linear model coeffs
    let coeffs = vand_a_pinv * y;
    // lin_coeffs.get(.., 0) is slopes
    // lin_coeffs.get(.., 1) is intercepts
    coeffs
}

/// Compute partial derivatives of dy/dx_i
/// from linear fit through the points
pub fn jac_from_lin(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64>
{
    let coeffs = linear_fit(x, y);
    let lin_coeffs = coeffs.get(0..x.ncols(), ..).transpose().to_owned();
    lin_coeffs
}

/// build a matrix with colums of powers of original maxtrix colums
/// up to degree max_degree
pub fn build_xpowers(x: MatRef<f64>, max_degree: usize) -> Mat<f64> {
    let mut x_out = x.to_owned();
    for deg in 2..max_degree {
        let x_new = mat_ele_pow(x, deg as f64);
        x_out = mat_hstack(x_out.as_ref(), x_new.as_ref());
    }
    x_out
}

/// Build vandermonde matrix
pub fn build_vandermonde(x: MatRef<f64>, include_self_interactions: bool) -> Mat<f64>
{
    let inter_x = mat_col_interactions(x, include_self_interactions);
    let all_x = mat_hstack(x, inter_x.as_ref());
    let mut ones_col = faer::Mat::zeros(x.nrows(), 1);
    ones_col.fill(1.0);
    let vand_a = mat_hstack(all_x.as_ref(), ones_col.as_ref());
    vand_a
}

/// Fit a quadratic in k dimensions
/// where k is the number of cols in x
pub fn quad_fit(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64>
{
    let vand_a = build_vandermonde(x, true);
    let vand_a_pinv = mat_pinv(vand_a.as_ref());
    let coeffs = vand_a_pinv * y;
    coeffs
}

/// Eval quadratic at x points
pub fn quad_eval(x: MatRef<f64>, coeffs: MatRef<f64>) -> Mat<f64>
{
    let vand_a = build_vandermonde(x, true);
    vand_a * coeffs
}

/// Compute partial derivatives of dy/dx_i at x0
/// from poly fit through point cloud
pub fn jac_from_quad(x0: MatRef<f64>, coeffs: MatRef<f64>) -> Mat<f64>
{
    let k_features = x0.ncols();
    let eps: f64 = 1.0e-10;
    let y0 = quad_eval(x0, coeffs.as_ref());
    let mut out: Mat<f64> = faer::Mat::zeros(x0.nrows(), x0.ncols());
    // for each feature, compute derivatives
    for k in 0..k_features {
        let mut x0_purt = x0.clone().to_owned();
        x0_purt.col_as_slice_mut(k)
            .into_iter()
            .for_each(|ele| *ele = *ele + eps);
        let y_purt = quad_eval(x0_purt.as_ref(), coeffs.as_ref());
        let dy_dxk = (y_purt - &y0) * faer::scale(1. / eps);
        out.col_as_slice_mut(k)
            .into_iter().enumerate()
            .for_each(|(i, ele)| *ele = dy_dxk.read(i, 0));
    }
    out
}



#[cfg(test)]
mod stats_corr_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_pearson() {
        // generate uncorrolated gaussian samples, should have cov
        // mat with 1 on diag
        let n_samples = 10000;
        let data_dim = 5;
        let x_tst = random_mat_normal::<f64>(n_samples, data_dim);

        let x_pcov = pearson_corr(x_tst.as_ref());
        let expect_pcov = faer::mat![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        print!("pearson coeff x: {:?}", x_pcov);
        mat_mat_approx_eq(
            x_pcov.as_ref(), expect_pcov.as_ref(), 1e-1f64);
    }

    #[test]
    fn test_cov() {
        // generate uncorrolated gaussian samples, should have cov
        // mat with 1 on diag
        let n_samples = 10000;
        let data_dim = 5;
        let x_tst = random_mat_normal::<f64>(n_samples, data_dim);

        let x_cov = mat_cov_centered(x_tst.as_ref());
        let expect_cov = faer::mat![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        print!("cov x: {:?}", x_cov);
        mat_mat_approx_eq(
            x_cov.as_ref(), expect_cov.as_ref(), 1e-1f64);
    }

    #[test]
    fn test_lin_fit() {
        // construct points along line with slope=0.5
        let x = faer::mat![
            [0.00, ],
            [0.25, ],
            [0.50, ],
            [0.75, ],
            [1.00, ],
        ];
        let y = faer::mat![
            [0.00 / 2.0, ],
            [0.25 / 2.0, ],
            [0.50 / 2.0, ],
            [0.75 / 2.0, ],
            [1.00 / 2.0, ],
        ];
        // fit the linear model
        let lin_coeffs = linear_fit(x.as_ref(), y.as_ref());
        // should have intercept 0, slope=0.5
        print!("lin_coefs x: {:?}", lin_coeffs);
        let jac_xy = jac_from_lin(x.as_ref(), y.as_ref());
        print!("jac xy: {:?}", jac_xy);
        let expected = faer::mat![[0.5]];
        mat_mat_approx_eq(jac_xy.as_ref(), expected.as_ref(), 1.0e-8f64);

        // contruct points along plane
        let xd = faer::mat![
            [0.00, 0.00],
            [1.00, 0.00],
            [0.00, 1.00],
            [1.00, 1.00],
        ];
        let yd = faer::mat![
            [0.00],
            [0.50],
            [0.50],
            [1.00],
        ];
        // fit the linear model
        let lin_coeffs_xd = linear_fit(xd.as_ref(), yd.as_ref());
        // should have intercept 0, slope=0.5
        print!("lin_coefs xd: {:?}", lin_coeffs_xd);
        let jac_xyd = jac_from_lin(xd.as_ref(), yd.as_ref());
        print!("jac xyd: {:?}", jac_xyd);
        let expected = faer::mat![[0.5, 0.5]];
        mat_mat_approx_eq(jac_xyd.as_ref(), expected.as_ref(), 1.0e-8f64);
    }

    #[test]
    fn test_quad_fit() {
        // contruct points
        let xd = faer::mat![
            [0.00, 0.00],
            [0.50, 0.00],
            [1.00, 0.00],
            [0.25, 0.25],
            [0.50, 0.50],
            [1.00, 1.00],
        ];
        let yd = faer::mat![
            [0.00],
            [0.25],
            [0.50],
            [0.30],
            [0.50],
            [1.00],
        ];
        let quad_coeffs = quad_fit(xd.as_ref(), yd.as_ref());
        let jac_xd = jac_from_quad(xd.as_ref(), quad_coeffs.as_ref());
        print!("quad jac xd: {:?}", jac_xd);
    }

    #[test]
    fn test_col_interact() {
        let tst_x = faer::mat![
            [1.00, 2.00, 3.0, 4.0],
            [1.00, 2.00, 3.0, 4.0],
            [1.00, 2.00, 3.0, 4.0],
            [1.00, 2.00, 3.0, 4.0],
        ];
        let res = mat_col_interactions(tst_x.as_ref(), true);
        print!("interactions x: {:?}", res);
        //   x1*x1 x1*x2 x1*x3 x1*x4 x2*x2 x2*x3 x2*x4 x3*x3 x3*x4  x4*x4
        let expected = mat![
            [1.00, 2.00, 3.00, 4.00, 4.00, 6.00, 8.00, 9.00, 12.00, 16.00],
            [1.00, 2.00, 3.00, 4.00, 4.00, 6.00, 8.00, 9.00, 12.00, 16.00],
            [1.00, 2.00, 3.00, 4.00, 4.00, 6.00, 8.00, 9.00, 12.00, 16.00],
            [1.00, 2.00, 3.00, 4.00, 4.00, 6.00, 8.00, 9.00, 12.00, 16.00],
            ];
        mat_mat_approx_eq(res.as_ref(), expected.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_rsquared_sens() {
        let tst_cov = faer::mat![
            [0.9, 0.5],
            [0.5, 0.9],
        ];
        let x_tst = sample_mv_normal(tst_cov.as_ref(), 100);
        print!("mv norm x: {:?}", x_tst);
        let mut y_tst = faer::Mat::zeros(x_tst.nrows(), 1);
        let tst_fn = |x1: f64, x2: f64| -> f64 {x1.powf(1.) + x2.powf(2.)};
        // evaluate the tst funciton at all samles
        for (i, sample) in x_tst.row_chunks(1).enumerate() {
            let ys: f64 = tst_fn(sample[(0, 0)], sample[(0, 1)]);
            y_tst.write(i, 0, ys);
        }
        // compute rsquared
        let tst_rsq = rsquared_sens(x_tst.as_ref(), y_tst.as_ref(), true);
        print!("rsqr x: {:?}", tst_rsq);
        assert!(tst_rsq.ncols() == 1);
        assert!(tst_rsq.nrows() == 1);
        assert!(tst_rsq.read(0, 0) < 1.);
        assert!(tst_rsq.read(0, 0) > 0.);
    }
}
