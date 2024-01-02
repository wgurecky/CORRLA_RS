use numpy::ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, pyclass, pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;

use faer::{mat, Mat, MatRef, IntoFaer, IntoNdarray};
use faer::{prelude::*};
use crate::lib_math_utils::space_samplers::*;
use crate::lib_math_utils::pod_rom::*;
use crate::lib_math_utils::interp_utils::*;
use crate::lib_math_utils::random_svd::*;
use crate::lib_math_utils::pca_rsvd::{PcaRsvd};
use crate::lib_math_utils::active_subspaces::{ActiveSsRsvd, PolyGradientEstimator};

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
        // let expl_var = my_pca.explained_var();
        let components = my_pca.components();
        let singular_vals = my_pca.singular_values();
        let ndarray_sv: Array2<f64> = singular_vals.as_ref().into_ndarray().to_owned();
        let ndarray_pc: Array2<f64> = components.as_ref().into_ndarray().to_owned();
        (ndarray_sv.into_pyarray(py), ndarray_pc.into_pyarray(py))
    }

    #[pyfn(m)]
    fn active_ss<'py>(py: Python<'py>, a_mat: PyReadonlyArray2<'py, f64>,
                      y: PyReadonlyArray2<'py, f64>,
                      order: usize, n_nbr: usize, n_comps: usize)
        -> (&'py PyArray2<f64>, &'py PyArray2<f64>)
    {
        // convert numpy array into rust ndarray and
        // convert to faer-rs matrix
        let x = a_mat.as_array();
        let x_mat = x.view().into_faer();
        let fx = y.as_array();
        let y_mat = fx.view().into_faer();

        // init gradient estimator
        let grad_est = PolyGradientEstimator::new(
            x_mat, y_mat, order, n_nbr);

        // compute active subspace directions and singular values
        let mut act_ss = ActiveSsRsvd::new(grad_est, n_comps);
        act_ss.fit(x_mat.as_ref());
        let components = act_ss.components_.unwrap();
        let singular_vals = act_ss.singular_vals_.unwrap();
        let ndarray_pc: Array2<f64> = components.as_ref().into_ndarray().to_owned();
        let ndarray_pv: Array2<f64> = singular_vals.as_ref().into_ndarray().to_owned();
        (ndarray_pc.into_pyarray(py), ndarray_pv.into_pyarray(py))
    }

    #[pyfn(m)]
    fn cs_dirichlet_sample<'py>(py: Python<'py>,
        np_bounds: PyReadonlyArray2<'py, f64>,
        n_samples: usize,
        max_zshots: usize,
        chunk_size: usize,
        c_scale: f64,
        ) -> &'py PyArray2<f64>
    {
        let bounds = np_bounds.as_array();
        let samples = constr_dirichlet_sample(
            bounds, n_samples, max_zshots, chunk_size, c_scale);
        let ndarray_samples = samples.to_owned();
        ndarray_samples.into_pyarray(py)
    }

    // Add classes to module
    m.add_class::<PyRbfInterp>()?;
    m.add_class::<PyPodI>()?;

    Ok(())
}

/// Python interface for rust RBF Interp impl
#[pyclass(unsendable)]
pub struct PyRbfInterp {
    pub rbfi: RbfInterp,
}

#[pymethods]
impl PyRbfInterp {
    #[new]
    pub fn new(kernel_type: usize, kernel_param: f64, dim: usize, poly_degree: usize) -> Self {
        let rbf_kern: Box<dyn RbfEval> = match kernel_type {
            1 => { Box::new(RbfKernelLin::new()) },
            2 => { Box::new(RbfKernelMultiQuad::new(kernel_param)) },
            3 => { Box::new(RbfKernelCubic::new()) },
            _ => { Box::new(RbfKernelGauss::new(kernel_param)) },
        };
        let rbf_interp_f = RbfInterp::new(rbf_kern, dim, poly_degree);
        Self {
            rbfi: rbf_interp_f,
        }
    }

    pub fn fit(&mut self, x_np: PyReadonlyArray2<f64>, y_np: PyReadonlyArray2<f64>) {
        // convert numpy array into rust ndarray and
        // convert to faer-rs matrix
        let x = x_np.as_array();
        let x_mat = x.view().into_faer();
        let y = y_np.as_array();
        let y_mat = y.view().into_faer();
        // fit the RBF interpolant
        self.rbfi.fit(x_mat.as_ref(), y_mat.as_ref());
    }

    pub fn predict(&self, py: Python<'_>, x_np: PyReadonlyArray2<f64>)
        -> Py<PyArray2<f64>>
    {
        let x = x_np.as_array();
        let x_mat = x.view().into_faer();
        let y_eval = self.rbfi.predict(x_mat.as_ref());
        let ndarray_y: Array2<f64> = y_eval.as_ref().into_ndarray().to_owned();
        ndarray_y.into_pyarray(py).to_owned()
    }
}

/// Python interface for rust POD Interp impl
#[pyclass(unsendable)]
pub struct PyPodI {
    pub pod: PodI,
}

#[pymethods]
impl PyPodI {
    #[new]
    pub fn new(x_np: PyReadonlyArray2<f64>, t_np: PyReadonlyArray2<f64>, n_modes: usize) -> Self
    {
        let x = x_np.as_array();
        let x_mat = x.view().into_faer();
        let t = t_np.as_array();
        let t_mat = t.view().into_faer();
        PyPodI {
            pod: PodI::new(x_mat.as_ref(), t_mat.as_ref(), n_modes),
        }
    }

    pub fn predict(&self, py:Python<'_>, t_np: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>>
    {
        let t = t_np.as_array();
        let t_mat = t.view().into_faer();
        let y_eval = self.pod.predict(t_mat.as_ref());
        let ndarray_y: Array2<f64> = y_eval.as_ref().into_ndarray().to_owned();
        ndarray_y.into_pyarray(py).to_owned()
    }
}
