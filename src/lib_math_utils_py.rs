use numpy::ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, pyclass, pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;

use faer::{mat, Mat, MatRef, IntoFaer, IntoNdarray};
use faer::{prelude::*};
use crate::lib_math_utils::dmd_rom::*;
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
        -> (&'py PyArray2<f64>, &'py PyArray2<f64>, &'py PyArray1<f64>)
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
        let fit_act_ss = ActiveSsRsvd::new(grad_est, n_comps)
            .fit(x_mat.as_ref());
        let components = fit_act_ss.components();
        let singular_vals = fit_act_ss.singular_vals();
        // compute sensitivity coeffs
        let var_sensi = fit_act_ss.var_diag_evd_sensi();
        // convert to python arrays
        let ndarray_pc: Array2<f64> = components.as_ref().into_ndarray().to_owned();
        let ndarray_pv: Array2<f64> = singular_vals.as_ref().into_ndarray().to_owned();
        (ndarray_pc.into_pyarray(py),
         ndarray_pv.into_pyarray(py),
         var_sensi.into_pyarray(py))
    }

    #[pyfn(m)]
    fn cs_dirichlet_sample<'py>(py: Python<'py>,
        np_bounds: PyReadonlyArray2<'py, f64>,
        n_samples: usize,
        max_zshots: usize,
        chunk_size: usize,
        c_scale: f64,
        np_alphas: PyReadonlyArray1<'py, f64>
        ) -> &'py PyArray2<f64>
    {
        let bounds = np_bounds.as_array();
        let vec_alphas = Some(np_alphas.as_array().to_vec());
        let samples = constr_dirichlet_sample(
            bounds, n_samples, max_zshots, chunk_size, c_scale, vec_alphas);
        let ndarray_samples = samples.to_owned();
        ndarray_samples.into_pyarray(py)
    }

    #[pyfn(m)]
    fn cs_mcmc_dirichlet_sample<'py>(py: Python<'py>,
        np_bounds: PyReadonlyArray2<'py, f64>,
        n_samples: usize,
        n_seed_samples: usize,
        max_zshots: usize,
        chunk_size: usize,
        c_scale: f64,
        np_alphas: PyReadonlyArray1<'py, f64>,
        gamma: f64,
        var_epsilon: f64,
        ) -> (&'py PyArray2<f64>, f64)
    {
        let bounds = np_bounds.as_array();
        let vec_alphas = Some(np_alphas.as_array().to_vec());
        // draw seed samples for second mcmc stage
        let diri_samples = constr_dirichlet_sample(
            bounds, n_seed_samples, max_zshots,
            chunk_size, c_scale, vec_alphas.clone());

        // Setup ln likelihood
        // alpha args to target dirichelt is 1, we want uniform samples in z
        let target_dirichlet_alphas = vec![1.0; vec_alphas.unwrap().len()];
        let tst_ln_like = LnLikeDirichlet::new(&target_dirichlet_alphas);
        // Setup ln prior
        let tst_ln_prior = LnPriorUniform::new(bounds.view());
        // construct likelihood*prior
        let tst_ln_like_prior = LnLikeSum::new(tst_ln_like, tst_ln_prior);
        // define fixup function as a lambda fn
        let proposal_fix_fn = move | x: ArrayView1<f64> | -> Array1<f64> {
            let mut new_x = x.to_owned();
            // must sum to c_scale (typically 1.0)
            new_x = c_scale * new_x.clone() / new_x.sum();
            new_x
        };
        let n_chains = diri_samples.nrows();
        // setup parallel mcmc chains for demc sampler
        let mut tst_chains: Vec<McmcChain> = Vec::new();
        for (c, seed_s) in diri_samples.rows().into_iter().enumerate() {
            println!("seed i: {:?}, x: {:?}, x_sum: {:?}", c, seed_s, seed_s.sum());
            tst_chains.push(McmcChain::new(3, seed_s, c));
        }
        // init the MCMC sampler
        let ndim = bounds.nrows();
        // var_epsilon typically 1.0e-12 (move jitter factor)
        // gamma typically 0.8 (move shrink factor)
        let mut mcmc_sampler =
            DeMcSampler::new(tst_ln_like_prior, tst_chains, ndim,
                             gamma, var_epsilon);
        // specify the proposal fixup function
        mcmc_sampler.set_prop_fixup(proposal_fix_fn);

        // draw samples
        mcmc_sampler.sample_mcmc_par(n_samples);
        // acceptance ratio
        let ar = mcmc_sampler.accept_ratio();

        // return samples
        let samples = mcmc_sampler.get_samples(n_samples);
        let ndarray_samples = samples.to_owned();
        (ndarray_samples.into_pyarray(py), ar)
    }

    // Add classes to module
    m.add_class::<PyRbfInterp>()?;
    m.add_class::<PyPodI>()?;
    m.add_class::<PyDMDc>()?;

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


/// Python interface for rust DMDc impl
#[pyclass(unsendable)]
pub struct PyDMDc {
    pub dmd: DMDc<'static>,
}

#[pymethods]
impl PyDMDc {
    #[new]
    pub fn new(x_np: PyReadonlyArray2<f64>, u_np: PyReadonlyArray2<f64>, n_modes: usize, n_iters: usize) -> Self
    {
        let x = x_np.as_array();
        let x_mat = x.view().into_faer();
        let u = u_np.as_array();
        let u_mat = u.view().into_faer();
        Self {
            dmd: DMDc::new(x_mat.as_ref(), u_mat.as_ref(), 1.0, n_modes, n_iters),
        }
    }

    pub fn predict(&self, py:Python<'_>, x0_np: PyReadonlyArray2<f64>, u_np: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>>
    {
        let x0 = x0_np.as_array();
        let x_mat = x0.view().into_faer();
        let u = u_np.as_array();
        let u_mat = u.view().into_faer();
        let y_eval = self.dmd.predict_multiple(x_mat.as_ref(), u_mat.as_ref());
        let ndarray_y: Array2<f64> = y_eval.as_ref().into_ndarray().to_owned();
        ndarray_y.into_pyarray(py).to_owned()
    }
}
