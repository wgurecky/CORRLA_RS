use numpy::ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyResult, Python};

use faer::{mat, Mat, MatRef, IntoFaer, IntoNdarray};
use faer::{prelude::*};
use crate::lib_math_utils::random_svd::*;
use crate::lib_math_utils::pca_rsvd::{PcaRsvd};

#[pymodule]
fn corrla_rs<'py>(_py: Python<'py>, m: &'py PyModule)
    -> PyResult<()>
{
    #[pyfn(m)]
    fn rsvd<'py>(py: Python<'py>, a_mat: PyReadonlyArray2<'py, f64>, n_rank: usize, n_iters: usize, n_oversamples: usize)
        -> (&'py PyArray2<f64>, &'py PyArray2<f64>, &'py PyArray2<f64>)
    {
        // convert numpy array into rust ndarray and
        // convert to faer-rs matrix
        let x = a_mat.as_array();
        let y = x.view().into_faer();
        // pass to rsvd routine
        let (ur, sr, vr) = random_svd(y, n_rank, n_iters, n_oversamples);
        // pass rsvd result back to python
        let ndarray_sr: Array2<f64> = sr.as_ref().into_ndarray().to_owned();
        let ndarray_ur: Array2<f64> = ur.as_ref().into_ndarray().to_owned();
        let ndarray_vr: Array2<f64> = vr.as_ref().into_ndarray().to_owned();
        (ndarray_ur.into_pyarray(py), ndarray_sr.into_pyarray(py), ndarray_vr.into_pyarray(py))
    }

    #[pyfn(m)]
    fn rpca<'py>(py: Python<'py>, a_mat: PyReadonlyArray2<'py, f64>, n_rank: usize, n_iters: usize, n_oversamples: usize)
        -> (&'py PyArray2<f64>, &'py PyArray2<f64>)
    {
        // convert numpy array into rust ndarray and
        // convert to faer-rs matrix
        let x = a_mat.as_array();
        let y = x.view().into_faer();

        // Compute PCA
        let my_pca = PcaRsvd::new(y.as_ref(), n_rank);
        let expl_var = my_pca.explained_var();
        let components = my_pca.components();
        let ndarray_ev: Array2<f64> = expl_var.as_ref().into_ndarray().to_owned();
        let ndarray_pc: Array2<f64> = components.as_ref().into_ndarray().to_owned();
        (ndarray_ev.into_pyarray(py), ndarray_pc.into_pyarray(py))
    }
    Ok(())
}