// Impl random SVD from:
use std::cmp;
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use rand::{prelude::*};
use rand_distr::{StandardNormal, Uniform};
use num_traits::Float;
use std::time::SystemTime;

// Internal imports
use crate::lib_math_utils::mat_utils::*;

/// From algorithm 9 in P. Martinsson, J. Tropp.
/// Randomized Numerical Linear Algebra:
/// Foundations & Algorithms.  arxiv.org/pdf/2002.01387.pdf
pub fn power_iter<T>(a_mat: MatRef<T>, omega_rank: usize, n_iter: usize)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    // Generate random matrix, omega
    let a_ncols = a_mat.ncols();
    let a_nrows = a_mat.nrows();
    // let omega = random_mat_normal::<T>(a_nrows, omega_rank);
    let omega = random_mat_normal::<T>(a_ncols, omega_rank);
    let o_ncols = omega.ncols();
    let o_nrows = omega.nrows();

    // initial guess for y_mat is product A*omega
    // a_mat is nxm, with n >> m. omega is mxk
    // y_mat is nxk
    let mut y_mat = a_mat * omega;

    // storage for matmul results
    let mut o_mat_res: Mat<T> = Mat::zeros(o_nrows, o_ncols);
    for i in 0..n_iter {
        // update y_mat qr
        if i > 2 {
            y_mat = y_mat.qr().compute_thin_q();
        }
        // y_mat = a_mat * (a_mat.transpose() * &y_mat);
        // parallel impl of above
        par_matmul_helper(
            o_mat_res.as_mut(),
            a_mat.transpose().as_ref(),
            y_mat.as_ref(), T::from(1.0).unwrap(),
            8);
        par_matmul_helper(
            y_mat.as_mut(),
            a_mat.as_ref(),
            o_mat_res.as_ref(), T::from(1.0).unwrap(),
            8);
        // apply norm
        // y_mat = y_mat.as_ref() * faer::scale(
        //    T::from(1.).unwrap() / y_mat.norm_l2()
        //    );
    }
    let my_q = y_mat.qr().compute_thin_q();
    my_q
}


/// Randomized SVD
pub fn random_svd<T>(a_mat: MatRef<T>, omega_rank: usize, n_iter: usize, n_oversamples: usize)
    -> (Mat<T>, Mat<T>, Mat<T>)
    where
    T: faer_core::RealField + Float
{
    // if matrix is fat, make thin
    let mut aa_mat = a_mat;
    let mut fat: bool = false;
    if a_mat.nrows() < a_mat.ncols() {
        fat = true;
        aa_mat = a_mat.transpose();
    }
    // get random projection of a_mat onto my_q
    let my_q_mat = power_iter(
        aa_mat, cmp::min(omega_rank+n_oversamples, aa_mat.ncols()), n_iter);
    // q_mat is nxk, q_mat.T is kxn  , aa_mat is nxm
    // b mat is kxm
    let my_b_mat = my_q_mat.transpose() * aa_mat;
    // let mut my_b_mat = faer::Mat::zeros(my_q_mat.ncols(), aa_mat.ncols());
    // par_matmul_helper(
    //     my_b_mat.as_mut(),
    //     my_q_mat.transpose().as_ref(),
    //     aa_mat.as_ref(),
    //     T::from(1.0).unwrap(), 8);

    // compute svd of reduced B matrix
    let my_rsvd = my_b_mat.svd();

    // map back to original space
    let u = my_q_mat * my_rsvd.u();

    // A = U * S * V
    // A^T = V^T * S^T * U^T
    if fat == true {
        (
         my_rsvd.v().transpose().get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         u.transpose().get(0..omega_rank, ..).to_owned(),
        )
    }
    else {
        (
         u.get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         my_rsvd.v().get(0..omega_rank, ..).to_owned()
        )
    }
}


#[cfg(test)]
mod rsvd_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_rsvd_f64() {
    // set global parallelism
    faer_core::set_global_parallelism(faer_core::Parallelism::Rayon(8));

    // create random matrix
    let sys_timer = SystemTime::now();
    let ti = sys_timer.elapsed().unwrap();
    print!("running rng... \n", );
    let test_a: Mat<f64> = Mat::from_fn(
        10000, 100,
        |_i, _j| { thread_rng().sample(StandardNormal) } );
    let tf = sys_timer.elapsed().unwrap();
    print!("done rng...{:?} s \n", (tf - ti).as_secs_f64());

    // print!("sigular values: {:?}", test_a);
    let (ur, sr, vr) = random_svd(test_a.as_ref(), 4, 8, 10);
    let tf2 = sys_timer.elapsed().unwrap();
    print!("done rsvd...{:?} s \n", (tf2 - tf).as_secs_f64());
    print!("sigular values: {:?}", sr);
    // print!("sigular vec: {:?}", ur);
    //
    let test_b = random_mat_uniform(5, 4, 0., 1.);
    print!("test b: {:?}", test_b);
    let mean_b = mat_mean(test_b.as_ref(), 1);
    print!("mean b: {:?}", mean_b);
    let centered_b = center_mat_col(test_b.as_ref());
    print!("Centered b: {:?}", centered_b);
    }
}
