// Impl random SVD from:
use std::cmp;
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use rand::{prelude::*};
use rand_distr::{StandardNormal, Uniform};
use num_traits::Float;
use std::time::SystemTime;
//use polars::frame::{DataFrame};
//use polars::frame::row::{Row};
//use polars_core::prelude::*;
//use polars_lazy::prelude::*;
use polars::prelude::*;

// Internal imports
use crate::lib_math_utils::mat_utils::*;


pub fn build_ymat<T>(a_mat: MatRef<T>, omega_rank: usize)
    -> (Mat<T>, Mat<T>)
    where
    T: faer_core::RealField + Float
{
    // Generate random matrix, omega
    let a_ncols = a_mat.ncols();
    // let omega = random_mat_normal::<T>(a_nrows, omega_rank);
    let omega = random_mat_normal::<T>(a_ncols, omega_rank);

    // initial guess for y_mat is product A*omega
    // a_mat is nxm, with n >> m. omega is mxk
    // y_mat is nxk
    let y_mat = a_mat * omega.as_ref();
    (y_mat, omega)
}

pub fn build_ymat_frame<T>(a_frame: &DataFrame, omega_rank: usize)
    -> (Mat<T>, Mat<T>)
    where
    T: faer_core::RealField + Float
{
    let a_ncols = a_frame.width();
    let omega = random_mat_normal::<T>(a_ncols, omega_rank);
    let y_mat = frame_matmul(a_frame, omega.as_ref());
    (y_mat, omega)
}

/// Performs out-of-core matrix-matrix A*B multiples where A
/// is in the polars dataframe format and B is ref to in-memory faer Mat.
pub fn frame_matmul<T>(a_frame: &DataFrame, b_mat: MatRef<T>)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    // iterate over dataframe rows
    let a_nrows = a_frame.height();
    let a_ncols = a_frame.width();
    let b_ncols = b_mat.ncols();

    // storage for output
    let mut out = faer::Mat::zeros(a_nrows, b_ncols);

    // perform out of core matmul row-wise in a (slow)
    for i in 0..a_nrows {
        // get one row at a time
        let tmp_row = a_frame.get_row(i).unwrap();
        // iterate over entries in each output column and b_mat col
        // sum over a rows
        for mut col_out in out.col_chunks_mut(1).into_iter() {
            for (j, row_val) in tmp_row.0.iter().enumerate() {
                let tmp_aval = row_val.try_extract::<T>().unwrap();
                let tmp_bval = b_mat.read(j, i);
                col_out.write(j, 0, tmp_aval * tmp_bval + col_out.read(j, 0));
            }
        }
    }
    out
}

/// Performs out-of-core matrix-matrix A*B where B
/// is in the polars dataframe format and A is in-memory.
pub fn frame_matmul_b<T>(a_mat: MatRef<T>, b_frame: &DataFrame)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    // iterate over dataframe rows
    let a_nrows = a_mat.nrows();
    let a_ncols = a_mat.ncols();
    let b_ncols = b_frame.width();

    // storage for output
    let mut out = faer::Mat::zeros(a_nrows, b_ncols);
    let mut col_tmp: Mat<T> = faer::Mat::zeros(a_ncols, 1);

    // iterate over the columns of the b_frame
    for (j, col) in b_frame.iter().enumerate() {
        let col_frame = col.clone().into_frame().lazy();
        // convert to faer
        let col_mat = faer::polars::polars_to_faer_f64(col_frame).unwrap();
        // convert type
        for i in 0..a_ncols {
            col_tmp.write(i, 0, T::from(col_mat.read(0, i)).unwrap());
        }
        // multiply by a_mat, forms the jth col of the output
        let out_col = a_mat.as_ref() * col_tmp.as_ref();
        for i in 0..a_ncols {
            out.write(i, j, col_tmp.read(i, 0));
        }
    }

    out
}

/// From algorithm 9 in P. Martinsson, J. Tropp.
/// Randomized Numerical Linear Algebra:
/// Foundations & Algorithms.  arxiv.org/pdf/2002.01387.pdf
pub fn power_iter<T>(a_mat: MatRef<T>, omega_rank: usize, n_iter: usize)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    // initial guess for y_mat is product A*omega
    // a_mat is nxm, with n >> m. omega is mxk
    // y_mat is nxk
    let (mut y_mat, omega) = build_ymat(a_mat, omega_rank);
    let o_ncols = omega.ncols();
    let o_nrows = omega.nrows();

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
    }
    let my_q = y_mat.qr().compute_thin_q();
    my_q
}

pub fn power_iter_frame<T>(a_frame: &DataFrame, omega_rank: usize, n_iter: usize)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    // initial guess for y_mat is product A*omega
    // a_mat is nxm, with n >> m. omega is mxk
    // y_mat is nxk
    let (mut y_mat, _omega) = build_ymat_frame::<T>(a_frame, omega_rank);

    // transpose of original dataframe
    let a_frame_t = a_frame.to_owned().transpose(None, None).unwrap();

    for i in 0..n_iter {
        // update y_mat qr
        if i > 2 {
            y_mat = y_mat.qr().compute_thin_q();
        }
        // y_mat = a_mat * (a_mat.transpose() * &y_mat);
        y_mat = frame_matmul(&a_frame, frame_matmul(&a_frame_t, y_mat.as_ref()).as_ref() );
    }
    let my_q = y_mat.qr().compute_thin_q();
    my_q
}


/// Randomized SVD from parqet file.
/// Useful when data is too large to fit into RAM. See
/// PcaRsvdParquet for a large data use case.
pub fn random_svd_frame<T>(a_frame: DataFrame, omega_rank: usize, n_iter: usize, n_oversamples: usize)
    -> (Mat<T>, Mat<T>, Mat<T>)
    where
    T: faer_core::RealField + Float
{
    let mut aa_frame = a_frame.to_owned();
    let mut fat: bool = false;
    if a_frame.height() < a_frame.width() {
        fat = true;
        aa_frame = aa_frame.transpose(None, None).unwrap();
    }
    // get random projection of a_mat onto my_q
    let my_q_mat = power_iter_frame::<T>(
        &aa_frame, cmp::min(omega_rank+n_oversamples, aa_frame.width()), n_iter);
    let my_b_mat = frame_matmul_b(my_q_mat.transpose(), &aa_frame);

    // compute svd of reduced B matrix
    let my_rsvd = my_b_mat.svd();

    // map back to original space
    let u = my_q_mat * my_rsvd.u();

    // A = U * S * V
    // A^T = V^T * S^T * U^T
    if fat == true {
        (
         my_rsvd.v().get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         u.transpose().get(0..omega_rank, ..).to_owned(),
        )
    }
    else {
        (
         u.get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         my_rsvd.v().get(.., 0..omega_rank).transpose().to_owned()
        )
    }
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

    // compute svd of reduced B matrix
    let my_rsvd = my_b_mat.svd();

    // map back to original space
    let u = my_q_mat * my_rsvd.u();

    // A = U * S * V
    // A^T = V^T * S^T * U^T
    if fat == true {
        (
         my_rsvd.v().get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         u.transpose().get(0..omega_rank, ..).to_owned(),
        )
    }
    else {
        (
         u.get(.., 0..omega_rank).to_owned(),
         my_rsvd.s_diagonal().get(0..omega_rank, ..).to_owned(),
         my_rsvd.v().get(.., 0..omega_rank).transpose().to_owned()
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
