/// Proper Orthogonal Decomp with interpolation impl.
/// Represents N-Dim functions as linear combinations of "modes", phi
/// y(x, t) = \sum_i w_i(t)*\phi_i(x)
///
/// Useful for constructing simple ROMs where 2- or 3-D scalar fields
/// change as a function of a few input variables.
/// Ex: estimate the pressure field as a function of angle of attack
/// using a few pre-computed CFD snapshots.  The POD model interpolates
/// the CFD data to obtain an estimated pressure field at any attack angle.
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};
use num_traits::Float;

// internal imports
use crate::lib_math_utils::mat_utils::*;
use crate::lib_math_utils::random_svd::*;
use crate::lib_math_utils::interp_utils::*;

// Stores persistent data for POD
pub struct PodI {
    // number of supplied data snapshots
    n_snapshots: usize,
    // number of POD modes to compute and store for reconstructing output
    n_modes: usize,
    // POD mode storage (similar to eigenvecs)
    modes: Mat<f64>,
    // POD mode weight storage (similar to eig vals)
    mode_weights: Mat<f64>,
    // Exogenous var vals corresponding to each snapshot
    // location of POD mode interpolation support points
    t_abscissa: Mat<f64>,
    // Mode weight interpolator function
    mode_weight_f_t: Vec<RbfInterp>,
}

impl PodI {
    pub fn new(x_data: MatRef<f64>, t: MatRef<f64>, n_modes: usize) -> Self {
        assert_eq!(t.nrows(), x_data.nrows());
        let modes = Self::_modes(x_data, n_modes);
        let weights = Self::_weights(modes.as_ref(), x_data, n_modes);
        let interp_f = Self::_mode_interp(t, weights.as_ref());
        PodI {
            n_snapshots: x_data.nrows(),
            t_abscissa: t.to_owned(),
            modes: modes,
            mode_weights: weights,
            n_modes: n_modes,
            mode_weight_f_t: interp_f,
        }
    }

    /// Computes POD modes by RSVD
    fn _modes(x_data: MatRef<f64>, n_modes: usize)
        -> Mat<f64>
    {
        let (_u, _s, v) = random_svd(x_data, n_modes, 10, 10);
        v.transpose().to_owned()
    }

    /// Computes POD mode weights by least squares
    fn _weights(modes: MatRef<f64>, x_data: MatRef<f64>, n_modes: usize)
        -> Mat<f64>
    {
        let modes_inv = mat_pinv(modes);
        let mut opt_wgts = faer::Mat::zeros(x_data.nrows(), n_modes);
        for (i, row_x) in x_data.row_chunks(1).enumerate() {
            // compute optimal weights that best reconstructs each snapshot
            let t_weights = modes_inv.as_ref() * row_x.transpose();
            // opt_wgts[(i, ..)] = t_weights;
            for j in 0..n_modes {
                opt_wgts.write(i, j, t_weights.read(j, 0));
            }
        }
        opt_wgts
    }

    /// Constructs POD mode weights interpolants
    fn _mode_interp(t: MatRef<f64>, weights: MatRef<f64>)
        -> Vec<RbfInterp>
    {
        assert_eq!(t.nrows(), weights.nrows());
        // for each weight, interpolate on t
        // let tw_vec: Vec::<RbfInterp> = weights.col_chunks(1).into_iter().map(
        let tw_vec: Vec::<RbfInterp> = weights.col_chunks(1).map(
            |t_wgt| {
                // t, t_wgt 1D interpolant
                let dim = t.ncols();
                let mut t_wgt_interp_f = RbfInterp::new(
                    Box::new(RbfKernelLin::new()), dim, 1);
                t_wgt_interp_f.fit(t, t_wgt);
                t_wgt_interp_f
            }
            ).collect();
        tw_vec
    }

    /// Refit the POD model
    pub fn fit(&mut self, x_data: MatRef<f64>, t: MatRef<f64>, n_modes: usize)
    {
        *self = Self::new(x_data, t, n_modes)
    }

    /// Predict at point, t
    /// y(t) = \sum_i w_i(t)*\phi_i
    /// where \phi_i are the POD modes and w_i are the interpolated
    /// POD mode weights at t.
    pub fn predict(&self, t_query: MatRef<f64>) -> Mat<f64>
    {
        assert_eq!(t_query.nrows(), 1);
        let mut w_mat: Mat<f64> = faer::Mat::zeros(self.n_modes, 1);
        for j in 0..self.n_modes {
            w_mat.write(j, 0, self.mode_weight_f_t[j].predict(t_query)[(0, 0)]);
        }
        // modes is Nxm
        // w_mat is mx1
        // output is Nx1
        self.modes.as_ref() * w_mat.as_ref()
    }
}


#[cfg(test)]
mod pod_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_pod() {
        // build pressure field snapshots
        let sigma = 0.25f64;
        let nx = 100;
        let x_points = mat_linspace::<f64>(0.0, 10.0, nx);
        let n_snapshots = 20;
        let t_points = mat_linspace::<f64>(1.0, 9.0, n_snapshots);

        let mut p_snapshots: Mat<f64> = faer::Mat::zeros(nx, n_snapshots);

        for n in 0..n_snapshots {
            let t = t_points[(n, 0)];
            let tmp_vec: Vec<_> = x_points.col_as_slice(0).into_iter().map(
                |x| { (0.5 * t) * (-(x-t).powf(2.0) / sigma.powf(2.0)).exp() }
            ).collect();
            let tmp_mat = mat_from_vec(&tmp_vec);
            mat_set_col(p_snapshots.as_mut(), n, tmp_mat.as_ref());
        }
        p_snapshots = p_snapshots.transpose().to_owned();

        // Compute POD modes
        let pod = PodI::new(p_snapshots.as_ref(), t_points.as_ref(), 4);

        // predict pressure at t=5.2
        let t_tst = faer::mat![[5.2,]];
        let pred_p = pod.predict(t_tst.as_ref());
        print!("Predicted P: {:?}", pred_p);
    }
}
