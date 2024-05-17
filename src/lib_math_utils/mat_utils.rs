/// Matrix utility methods
/// Contains helper methods that would normally be
/// found in a high level matrix and linear algebra
/// library, but are generally missing from rust libs
///
use num_traits::Float;
use faer::{prelude::*, IntoNdarray};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use faer_core::ColRef;
use faer_core::{c64, c32};
use rand::{prelude::*};
use rand_distr::{StandardNormal, Uniform};
use rayon::prelude::*;
use ndarray::prelude::*;


/// Matrix multiplication with explicit
/// parallel control exposed
pub fn par_matmul_helper<T>(res: MatMut<T>, lhs: MatRef<T>, rhs: MatRef<T>, beta: T, n_threads: usize)
    where
    T: faer_core::RealField + Float
{
    faer_core::mul::matmul(
        res,
        lhs,
        rhs,
        None,
        beta,
        // faer_core::Parallelism::Rayon(n_threads),
        faer_core::get_global_parallelism()
        );
}


/// Compute the Moore-Penrose inverse
pub fn mat_pinv<T>(x: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let x_svd = x.svd();
    let u = x_svd.u();
    let eps = T::from(1.0e-14).unwrap();
    let s_vec = x_svd.s_diagonal();
    let v = x_svd.v();
    let mut s_inv_mat = faer::Mat::zeros(x.ncols(), x.nrows());
    for i in 0..s_vec.nrows(){
        s_inv_mat.write(i, i, T::from(1.).unwrap() /
                        (s_vec.read(i, 0) + eps));
    }
    v * s_inv_mat * u.transpose()
}

/// Compute the Moore-Penrose inverse for complex
pub fn mat_pinv_comp(x: MatRef<c64>) -> Mat<c64>
{
    let x_svd = x.svd();
    let u = x_svd.u();
    let eps = c64::new(1.0e-16, 1.0e-16);
    let comp_one = c64::new(1.0, 0.0);
    let s_vec = x_svd.s_diagonal();
    let v = x_svd.v();
    let mut s_inv_mat: Mat<c64> = faer::Mat::zeros(x.ncols(), x.nrows());
    for i in 0..s_vec.nrows(){
        s_inv_mat.write(i, i, comp_one /
                        (s_vec.read(i, 0) + eps));
    }
    v * s_inv_mat * u.adjoint()
}

/// Truncated SVD
pub fn mat_truncated_svd<T>(my_mat: MatRef<T>, rank: usize) -> (Mat<T>, Mat<T>, Mat<T>)
    where
    T: faer_core::RealField + Float
{
    let my_svd = my_mat.svd();
    let u_r = my_svd.u().get(.., 0..rank).to_owned();
    let s_r = my_svd.s_diagonal().get(0..rank, ..).to_owned();
    let v_r = my_svd.v().get(.., 0..rank).to_owned();
    (u_r, s_r, v_r)
}


/// Computes mean along axis
pub fn mat_mean<T>(a_mat: MatRef<T>, axis: usize) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut acc: T = T::from(0.0).unwrap();
    let a_ncols: usize = a_mat.ncols();
    let a_nrows: usize = a_mat.nrows();

    match axis {
        0 => {
            let mut out_mat: Mat<T> = Mat::zeros(a_nrows, 1);
            for i in 0..a_nrows {
                acc = T::faer_zero();
                for j in 0..a_ncols {
                    acc = acc + a_mat.read(i, j);
                }
                out_mat.write(i, 0, acc / T::from(a_ncols).unwrap());
            }
            return out_mat
        }
        _ => {
            let mut out_mat: Mat<T> = Mat::zeros(1, a_ncols);
            for j in 0..a_ncols {
                acc = T::faer_zero();
                for i in 0..a_nrows {
                    acc = acc + a_mat.read(i, j);
                }
                out_mat.write(0, j, acc / T::from(a_nrows).unwrap());
            }
            return out_mat
        }
    }
}

/// Computes std dev along axis
pub fn mat_std<T>(a_mat: MatRef<T>, axis: usize) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut acc: T = T::from(0.0).unwrap();
    let a_ncols: usize = a_mat.ncols();
    let a_nrows: usize = a_mat.nrows();
    let mu = mat_mean(a_mat, axis);

    match axis {
        0 => {
            let mut out_mat: Mat<T> = Mat::zeros(a_nrows, 1);
            for i in 0..a_nrows {
                acc = T::faer_zero();
                for j in 0..a_ncols {
                    acc = acc + (a_mat.read(i, j) - mu.read(i, 0)).powf(
                        T::from(2.).unwrap());
                }
                out_mat.write(i, 0, (acc / T::from(a_ncols-1).unwrap()).sqrt());
            }
            return out_mat
        }
        _ => {
            let mut out_mat: Mat<T> = Mat::zeros(1, a_ncols);
            for j in 0..a_ncols {
                acc = T::faer_zero();
                for i in 0..a_nrows {
                    acc = acc + (a_mat.read(i, j) - mu.read(0, j)).powf(
                        T::from(2.).unwrap());
                }
                out_mat.write(0, j, (acc / T::from(a_nrows-1).unwrap()).sqrt());
            }
            return out_mat
        }
    }
}


/// create a matrix filled with standard normal samples
pub fn random_mat_normal<T>(n_rows: usize, n_cols: usize)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let omega: Mat<T> = Mat::from_fn(
        n_rows,
        n_cols,
        |_i, _j| {
            T::from::<f64>(
            thread_rng().sample(StandardNormal)).unwrap()
            }
        );
    omega
}

/// create a matrix filled with uniform random samples
pub fn random_mat_uniform<T>(n_rows: usize, n_cols: usize, lb: f64, ub: f64)
    -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let uni_dist = Uniform::new(lb, ub);
    let omega: Mat<T> = Mat::from_fn(
        n_rows,
        n_cols,
        |_i, _j| {
            T::from::<f64>(
            thread_rng().sample(uni_dist)).unwrap()
            }
        );
    omega
}


/// Applies a scalar function to each element of x
/// returns a new matrix with each element modified
/// by the fn fn_x
/// pub fn scalar_fn_mat(x: MatRef<f64>, fn_x: &dyn FnMut(f64)->f64)
pub fn mat_scalar_fn(x: MatRef<f64>, fn_x: &dyn Fn(f64)->f64)
    -> Mat<f64>
{
    let mut out_mat: Mat<f64> = x.to_owned();
    for j in 0..out_mat.ncols() {
        out_mat.col_as_slice_mut(j)
            .into_iter()
            .for_each(|ele| *ele = fn_x(*ele));
    }
    out_mat
}

/// Computes matrix*av av is a vec along axis
/// mul vec row by row or col by col
// pub fn mat_vec_mul(x: MatRef<f64>, av: MatRef<f64>, axis: usize)
//     -> Mat<f64>
// {
// }

/// Adds entries of a 1d vec to each column
pub fn mat_vec_col_add(a_mat: MatRef<f64>, in_vec: MatRef<f64>) -> Mat<f64>
{
    let mut out_mat: Mat<f64> = a_mat.to_owned();
    for j in 0..out_mat.ncols() {
        out_mat.col_as_slice_mut(j)
            .into_iter()
            .for_each(|ele| *ele = *ele + in_vec.read(0, j));
    }
    out_mat
}


/// Adds entries of a 1d vec to each row
pub fn mat_vec_row_add(a_mat: MatRef<f64>, in_vec: MatRef<f64>) -> Mat<f64>
{
    let mut out_mat: Mat<f64> = a_mat.transpose().to_owned();
    for i in 0..out_mat.ncols() {
        out_mat.col_as_slice_mut(i)
            .into_iter()
            .for_each(|ele| *ele = *ele + in_vec.read(i, 0));
    }
    out_mat.transpose().to_owned()
}

/// Computes matrix+pv av is a vec along axis
/// add vec row by row or col by col
pub fn mat_vec_add(x: MatRef<f64>, pv: MatRef<f64>, axis: usize)
    -> Mat<f64>
{
    match axis {
        0 => {
            mat_vec_col_add(x, pv)
        }
        _ => {
            mat_vec_row_add(x, pv)
        }
    }
}

/// Raise all elements in a matrix to a power
pub fn mat_ele_pow<T>(a_mat: MatRef<T>, pwr: T) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut out_mat = a_mat.to_owned();
    for i in 0..out_mat.nrows() {
        for j in 0..out_mat.ncols() {
            out_mat.write(i, j, out_mat.read(i, j).powf(T::from(pwr).unwrap()));
        }
    }
    out_mat
}

/// Element by element matrix multiplication
pub fn mat_mat_ele_mul<T>(a: MatRef<T>, b: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut out_mat = a.to_owned();
    for i in 0..out_mat.nrows() {
        for j in 0..out_mat.ncols() {
            out_mat.write(i, j, out_mat.read(i, j) * b.read(i, j));
        }
    }
    out_mat
}

/// Peforms matrix scalar addition in-place
pub fn mat_scalar_add<T>(mut out_mat: MatMut<T>, b: T)
    where
    T: faer_core::RealField + Float
{
    // consume a, modify in-place
    for i in 0..out_mat.nrows() {
        for j in 0..out_mat.ncols() {
            out_mat.write(i, j, out_mat.read(i, j) + b);
        }
    }
}

/// Modifies matrix row in-place
pub fn mat_row_mod<T>(mut out_mat: MatMut<T>, row: usize, vec: MatRef<T>)
    where
    T: faer_core::RealField + Float
{
    for j in 0..vec.ncols() {
        out_mat.write(row, j, vec.read(0, j));
    }
}

/// Modifies matrix col in-place
pub fn mat_col_mod<T>(mut out_mat: MatMut<T>, col: usize, vec: MatRef<T>)
    where
    T: faer_core::RealField + Float
{
    for i in 0..vec.nrows() {
        out_mat.write(i, col, vec.read(i, 0));
    }
}

/// combine real and imag components into complex mat
pub fn mat_complex_from_parts(a_re: MatRef<f64>, a_im: MatRef<f64>) -> Mat<c64> {
    let mut out_mat: Mat<c64> = faer::Mat::zeros(a_re.nrows(), a_re.ncols());
    for i in 0..out_mat.nrows() {
        for j in 0..out_mat.ncols() {
            out_mat.write(i, j, c64::new(a_re.read(i, j), a_im.read(i, j)));
        }
    }
    out_mat
}

/// split real and imag components from complex mat
pub fn mat_parts_from_complex(a_comp: MatRef<c64>) -> (Mat<f64>, Mat<f64>) {
    let mut re_mat: Mat<f64> = faer::Mat::zeros(a_comp.nrows(), a_comp.ncols());
    let mut im_mat: Mat<f64> = faer::Mat::zeros(a_comp.nrows(), a_comp.ncols());
    for i in 0..re_mat.nrows() {
        for j in 0..re_mat.ncols() {
            re_mat.write(i, j, a_comp.read(i, j).re);
            im_mat.write(i, j, a_comp.read(i, j).im);
        }
    }
    (re_mat, im_mat)
}


/// converts a col vector to a diagnoal matrix
pub fn mat_colvec_to_diag<T>(vec: MatRef<T>) -> Mat<T>
    where
    T: faer_core::ComplexField
{
    let mut out_mat = faer::Mat::zeros(vec.nrows(), vec.nrows());
    for i in 0..vec.nrows() {
        out_mat.write(i, i, vec.read(i, 0));
    }
    out_mat
}

/// converts a col mat to a diagnoal matrix
pub fn mat_colmat_to_diag<T>(vec: ColRef<T>) -> Mat<T>
    where
    T: faer_core::ComplexField
{
    let mut out_mat = faer::Mat::zeros(vec.nrows(), vec.nrows());
    for i in 0..vec.nrows() {
        out_mat.write(i, i, vec.read(i));
    }
    out_mat
}


/// converts a row vector to a diagnoal matrix
pub fn mat_rowvec_to_diag<T>(vec: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut out_mat = faer::Mat::zeros(vec.ncols(), vec.ncols());
    for i in 0..vec.ncols() {
        out_mat.write(i, i, vec.read(0, i));
    }
    out_mat
}

/// pseudo inv of diag matrix
pub fn mat_pinv_diag<T>(diag_mat: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let eps = T::from(1.0e-20).unwrap();
    let mut out_mat = faer::Mat::zeros(diag_mat.nrows(), diag_mat.ncols());
    for i in 0..diag_mat.ncols() {
        let tmp_val = diag_mat.read(i, i);
        if tmp_val < eps && tmp_val > -eps {
            out_mat.write(i, i, T::from(0.0).unwrap());
        }
        else {
            out_mat.write(i, i, T::from(1.0).unwrap() / (tmp_val + eps));
        }
    }
    out_mat
}

/// create a owned Vec from row, results in data copy
pub fn mat_row_to_vec<T>(a_mat: MatRef<T>, row: usize) -> Vec<T>
    where
    T: faer_core::SimpleEntity
{
    let tmp_ndarray = a_mat.get(row..row+1, ..).into_ndarray();
    let tmp_1darray: Array1<T> = tmp_ndarray.remove_axis(Axis(0)).to_owned();
    tmp_1darray.to_vec()
}

/// create a owned Vec from col, results in data copy
pub fn mat_col_to_vec<T>(a_mat: MatRef<T>, col: usize) -> Vec<T>
    where
    T: faer_core::SimpleEntity
{
    let tmp_ndarray = a_mat.get(.., col..col+1).into_ndarray();
    let tmp_1darray: Array1<T> = tmp_ndarray.remove_axis(Axis(1)).to_owned();
    tmp_1darray.to_vec()
}

/// create a owned Vec from mat diag, results in data copy
pub fn mat_diag_to_vec<T>(a_mat: MatRef<T>) -> Vec<T>
    where
    T: faer_core::SimpleEntity
{
    let mut diag_vec: Vec<T> = Vec::new();
    for i in 0..a_mat.ncols() {
        diag_vec.push(a_mat.read(i, i));
    }
    diag_vec
}

/// Returns indicies that would sort a vec of floats
pub fn argsort_float<T>(a: &[T]) -> Vec<usize>
    where
    T: faer_core::RealField + Float
{
    let mut idx = (0..a.len()).collect::<Vec<_>>();
    idx.sort_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap());
    idx
}

/// Centers the matrix such that the cols have zero mean.
/// returns a new matrix, does not modify original mat
pub fn center_mat_col(a_mat: MatRef<f64>) -> Mat<f64>
{
    let mut out_mat: Mat<f64> = a_mat.to_owned(); // clone happens here
    let col_avgs = mat_mean(a_mat.as_ref(), 1);
    for j in 0..out_mat.ncols() {
        // for ele in out_mat.col_as_slice_mut(j){
        //     *ele = *ele - col_avgs.read(0, j);
        // }
        // shorthand for above using closure
        // to modify elements of mutable iterator in-place
        out_mat.col_as_slice_mut(j)
            .into_iter()
            .for_each(|ele| *ele = *ele - col_avgs.read(0, j));
        // same as above, but collect result into vec and no modification in-place
        // let col_r: Vec::<_> = out_mat.col_as_slice_mut(j)
        //     .into_iter()
        //     .map(|ele| *ele = *ele - col_avgs.read(0, j))
        //     .collect();
    }
    out_mat
}


/// Centers the matrix such that the cols have zero mean and unit std
/// returns a new matrix, does not modify original mat
pub fn zcenter_mat_col(a_mat: MatRef<f64>) -> Mat<f64>
{
    let mut out_mat: Mat<f64> = a_mat.to_owned(); // clone happens here
    let col_avgs = mat_mean(a_mat, 1);
    let col_stds = mat_std(a_mat, 1);
    // parallel iterator over cols
//     out_mat.par_col_chunks_mut(1).enumerate()
//         .for_each(|(j, mut col)|
//                   for i in 0..col.nrows() {
//                       col[(i, 0)] =
//                         (col[(i, 0)] - col_avgs.read(0, j)) / col_stds.read(0, j);
//                   }
//                  );
    for j in 0..out_mat.ncols() {
        // for ele in out_mat.col_as_slice_mut(j){
        //     *ele = (*ele - col_avgs.read(0, j)) / col_stds.read(0, j);
        // }
        // shorthand for above using closure
        // to modify elements of mutable iterator in-place
        out_mat.col_as_slice_mut(j)
            .into_iter()
            .for_each(|ele| *ele = (*ele - col_avgs.read(0, j))
                      / col_stds.read(0, j));
    }
    out_mat
}


// Helper function to ensure two matrix are almost equal
pub fn mat_mat_approx_eq<T>(a: MatRef<T>, b: MatRef<T>, tol: T)
    where
    T: faer_core::RealField + Float
{
    use assert_approx_eq::assert_approx_eq;
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(a.nrows(), b.nrows());
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_approx_eq!(a.read(i, j), b.read(i, j), tol);
        }
    }
}


// Helper function to ensure all matrix ele are close to const
pub fn mat_scale_approx_eq<T>(a: MatRef<T>, b: T, tol: T)
    where
    T: faer_core::RealField + Float
{
    use assert_approx_eq::assert_approx_eq;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_approx_eq!(a.read(i, j), b, tol);
        }
    }
}


// Stack two matrix together, horizontally
pub fn mat_hstack<T>(a: MatRef<T>, b: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    assert!(a.nrows() == b.nrows());
    let o_nrows = a.nrows();
    let o_ncols = a.ncols() + b.ncols();
    let col_offset = a.ncols();
    let mut o_mat: faer::Mat<T> = faer::Mat::zeros(o_nrows, o_ncols);
    for i in 0..o_nrows {
        for j in 0..o_ncols {
            if j < a.ncols() {
                o_mat.write(i, j, a.read(i, j));
            }
            else {
                o_mat.write(i, j, b.read(i, j - col_offset));
            }
        }
    }
    o_mat
}


// Stack two matrix together, vertically
pub fn mat_vstack<T>(a: MatRef<T>, b: MatRef<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    assert!(a.ncols() == b.ncols());
    let o_nrows = a.nrows() + b.nrows();
    let o_ncols = a.ncols();
    let row_offset = a.nrows();
    let mut o_mat: faer::Mat<T> = faer::Mat::zeros(o_nrows, o_ncols);
    for i in 0..o_nrows {
        for j in 0..o_ncols {
            if i < a.nrows() {
                o_mat.write(i, j, a.read(i, j));
            }
            else {
                o_mat.write(i, j, b.read(i - row_offset, j));
            }
        }
    }
    o_mat
}

/// Produces a single column vector of elements evenly spaced
pub fn mat_linspace<T>(start: T, end: T, n_steps: usize) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut o_mat = faer::Mat::zeros(n_steps, 1);
    let delta = (end - start) / (T::from(n_steps).unwrap());
    for i in 0..n_steps {
        o_mat.write(i, 0, T::from(i).unwrap()*delta);
    }
    o_mat
}

/// Set a column inside input matrix to values in col_mat
/// Modifies the input matrix in-place
pub fn mat_set_col<T>(mut a_mat: MatMut<T>, col: usize, col_mat: MatRef<T>)
    where
    T: faer_core::RealField + Float
{
    for i in 0..col_mat.nrows() {
        a_mat.write(i, col, col_mat.read(i, 0) );
    }
}

/// Converts a rust Vec into a faer Mat.  Done by a data copy (can be expensive)
pub fn mat_from_vec<T>(in_vec: &Vec<T>) -> Mat<T>
    where
    T: faer_core::RealField + Float
{
    let mut o_mat: Mat<T> = faer::Mat::zeros(in_vec.len(), 1);
    for (i, ele) in in_vec.into_iter().enumerate() {
        o_mat.write(i, 0, *ele);
    }
    o_mat
}


#[cfg(test)]
mod mat_utils_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_par_matmul_mat_vec() {
        let a_tst = faer::mat![
            [1.0, 0.0],
            [0.0, 1.0],
        ];
        let b_tst = faer::mat![
            [3.0],
            [2.0],
        ];
        let mut out = faer::Mat::zeros(2, 1);
        // compute mat vec product in parallel
        par_matmul_helper(
            out.as_mut(),
            a_tst.as_ref(),
            b_tst.as_ref(),
            1.0, 2);
        let expected = faer::mat![[3.0], [2.0]];
        mat_mat_approx_eq(
            out.as_ref(), expected.as_ref(), 1e-6f64);
    }

    #[test]
    fn test_par_matmul_mat_mat() {
        let a_tst = faer::mat![
            [1.0, 0.0],
            [0.0, 1.0],
        ];
        let b_tst = faer::mat![
            [3.0, 0.0],
            [2.0, 0.0],
        ];
        let mut out = faer::Mat::zeros(2, 2);
        // compute mat vec product in parallel
        par_matmul_helper(
            out.as_mut(),
            a_tst.as_ref(),
            b_tst.as_ref(),
            1.0, 2);
        let expected = faer::mat![[3.0, 0.0], [2.0, 0.0]];
        mat_mat_approx_eq(
            out.as_ref(), expected.as_ref(), 1e-6f64);
    }

    #[test]
    fn test_matrix_ops() {
        let n_samples = 10000;
        let data_dim = 12;
        let x_tst = random_mat_normal::<f64>(n_samples, data_dim);

        // compute mean of columns
        let mean_x = mat_mean(x_tst.as_ref(), 1);
        print!("mean x: {:?}", mean_x);
        // mean should be 0
        mat_scale_approx_eq(mean_x.as_ref(), 0.0, 1.0e-1f64);

        // compute std dev of columns
        let std_x = mat_std(x_tst.as_ref(), 1);
        print!("std x: {:?}", std_x);
        // std should be 1
        mat_scale_approx_eq(std_x.as_ref(), 1.0, 1.0e-1f64);
    }

    #[test]
    fn test_center() {
        let n_samples = 20;
        let data_dim = 4;
        let x_tst = random_mat_normal::<f64>(n_samples, data_dim);

        let centered_x = center_mat_col(x_tst.as_ref());
        let mean_xc = mat_mean(centered_x.as_ref(), 1);
        mat_scale_approx_eq(mean_xc.as_ref(), 0.0, 1e-12f64);
    }

    #[test]
    fn test_zcenter() {
        let n_samples = 20;
        let data_dim = 4;
        let x_tst = random_mat_normal::<f64>(n_samples, data_dim);

        let centered_x = center_mat_col(x_tst.as_ref());
        let zcentered_x = zcenter_mat_col(x_tst.as_ref());
        print!("centered x: {:?}", centered_x);
        print!("zcentered x: {:?}", zcentered_x);
        let mean_xz = mat_mean(zcentered_x.as_ref(), 1);
        let std_xz = mat_std(zcentered_x.as_ref(), 1);
        print!("zcentered mean x: {:?}", mean_xz);
        print!("zcentered std x: {:?}", std_xz);
        mat_scale_approx_eq(mean_xz.as_ref(), 0.0, 1e-12f64);
        mat_scale_approx_eq(std_xz.as_ref(), 1.0, 1e-12f64);
    }

    #[test]
    fn test_pinv() {
        let x_tst = faer::mat![
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
        ];
        let x_pinv = mat_pinv(x_tst.as_ref());
        let expected = faer::mat![
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],];
        print!("pinv x: {:?}", x_pinv);
        mat_mat_approx_eq(x_pinv.as_ref(), expected.as_ref(), 1.0e-6f64);

        let x_tst = faer::mat![
            [4., 0.],
            [3., -5.],
        ];
        let x_pinv = mat_pinv(x_tst.as_ref());
        let expected = faer::mat![
            [0.25,  0.0],
            [0.15, -0.2],];
        print!("pinv x: {:?}", x_pinv);
        mat_mat_approx_eq(x_pinv.as_ref(), expected.as_ref(), 1.0e-6f64);
    }

    #[test]
    fn test_hstack() {
        let a_tst = faer::mat![
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],];
        let b_tst = faer::mat![
            [2.0],
            [2.0],
            [2.0],
            [2.0],];
        let hstacked = mat_hstack(a_tst.as_ref(), b_tst.as_ref());
        print!("c hstck: {:?}", hstacked);
        let expected = faer::mat![
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],];
        mat_mat_approx_eq(hstacked.as_ref(), expected.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_vstack() {
        let a_tst = faer::mat![
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],];
        let b_tst = faer::mat![
            [2.0, 3.0],
            ];
        let vstacked = mat_vstack(a_tst.as_ref(), b_tst.as_ref());
        print!("c hstck: {:?}", vstacked);
        let expected = faer::mat![
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [2.0, 3.0],];
        mat_mat_approx_eq(vstacked.as_ref(), expected.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_mat_power() {
        let a_tst = faer::mat![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],];
        let res = mat_ele_pow(a_tst.as_ref(), 2.0);
        print!("mat pwr: {:?}", res);
        let expected = faer::mat![
            [1.0, 4.0],
            [1.0, 4.0],
            [1.0, 9.0],
            [4.0, 16.0],];
        mat_mat_approx_eq(res.as_ref(), expected.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_mat_mat_ele_mul() {
        let a_tst = faer::mat![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],];
        let b_tst = faer::mat![
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],];
        let res = mat_mat_ele_mul(a_tst.as_ref(), b_tst.as_ref());
        print!("mat mat ele mul: {:?}", res);
        let expected = faer::mat![
            [1.0, 4.0],
            [1.0, 4.0],
            [1.0, 9.0],
            [4.0, 16.0],];
        mat_mat_approx_eq(res.as_ref(), expected.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_mat_colvec_to_diag() {
        let a_tst = faer::mat![
            [1.0],
            [2.0],
            [3.0],
            [4.0],];
        let a_diag_mat = mat_colvec_to_diag(a_tst.as_ref());
        let expected = faer::mat![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],];
        mat_mat_approx_eq(expected.as_ref(), a_diag_mat.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_mat_rowvec_to_diag() {
        let a_tst = faer::mat![
            [1.0, 2.0, 3.0, 4.0],];
        let a_diag_mat = mat_rowvec_to_diag(a_tst.as_ref());
        let expected = faer::mat![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],];
        mat_mat_approx_eq(expected.as_ref(), a_diag_mat.as_ref(), 1.0e-12f64);
    }

    #[test]
    fn test_mat_pinv_diag() {
        let a_mat = faer::mat![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],];
        let inv_a = mat_pinv_diag(a_mat.as_ref());
        let expected = faer::mat![
            [1.0 / 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / 4.0],];
        mat_mat_approx_eq(expected.as_ref(), inv_a.as_ref(), 1.0e-12f64);

        let a_mat = faer::mat![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],];
        let inv_a = mat_pinv_diag(a_mat.as_ref());
        let expected = faer::mat![
            [1.0 / 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0,       0.0],];
        mat_mat_approx_eq(expected.as_ref(), inv_a.as_ref(), 1.0e-12f64);
    }
}
