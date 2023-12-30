// Principle component analysis (PCA) using
// randomized svd methods.
use std::cmp;
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef, MatMut, Entity, AsMatRef, AsMatMut};

// internal imports
use crate::lib_math_utils::mat_utils::*;
use crate::lib_math_utils::random_svd::*;


/// Stores persistent data for pca methods.
pub struct PcaRsvd {
    // number of principle components to compute and store
    pca_rank: usize,
    // means of data columns
    means: Mat<f64>,
    // owned storage for computed principle vectors
    pca_u: Option<Mat<f64>>,
    // owned storage for computed principle vector weights (importances)
    pca_s: Mat<f64>,
    // pca component directions
    components_: Mat<f64>,
    n_samples: usize,
}


// Trait for hitting a target matrix with a linear operator
// that projects the target matrix onto the PCA basis vectors.
// This is done for dimensionality reduction of data.
pub trait ApplyTransform {
    // Defines apply_tr function signature which all types that impl the ApplyTransform
    // trait must implement and conform to
    fn apply_tr(&self, targ_mat: MatRef<f64>) -> Mat<f64>;

    // Maps from reduced space back to original space
    fn apply_inv_tr(&self, red_mat: MatRef<f64>) -> Mat<f64>;
}


impl ApplyTransform for PcaRsvd {
    /// Implements the ApplyTransform trait for the PcaRsvd type
    fn apply_tr(&self, targ_mat: MatRef<f64>) -> Mat<f64> {
        let centered_targ_mat = center_mat_col(targ_mat);
        centered_targ_mat * self.components_.transpose()
    }

    /// Implements inverse map for the PcaRsvd type
    fn apply_inv_tr(&self, red_mat: MatRef<f64>) -> Mat<f64> {
        mat_vec_col_add(
            (red_mat * self.components_.as_ref()).as_ref(), self.means.as_ref())
    }
}

impl PcaRsvd {
    pub fn new(x_mat: MatRef<f64>, rank: usize)
    -> Self
    {
        // set mean of all cols to 0
        let means = mat_mean(x_mat, 1);
        let n_samples = x_mat.nrows();
        let n_dim = x_mat.ncols();
        let cx_mat = center_mat_col(x_mat.as_ref());
        // compute svd of centered x_mat
        let (_ur, sr, vr) = random_svd(
            cx_mat.as_ref(), rank, 20, cmp::min(n_dim, 10));
        // the singular values
        let pca_s = sr;
        // self.pca_u = ur;
        // right eigenvectors, equal to eigvec(x_T * x)
        let components_ = vr;

        let mut pca_rsvd_inst = PcaRsvd {
            pca_rank: rank,
            means: means,
            pca_u: None,
            pca_s: pca_s,
            components_: components_,
            n_samples: n_samples,
        };
        pca_rsvd_inst
    }

    /// Updates the pca_basis and pca_weights
    pub fn fit(&mut self, x_mat: MatRef<f64>, rank: usize)
    {
        *self = Self::new(x_mat, rank)
    }

    /// explained varience per component
    pub fn explained_var(&self) -> Mat<f64>
    {
        // equal to eigvals(x_T * x), var explained
        let mut out_ev = self.pca_s.clone();
        out_ev.col_as_slice_mut(0).into_iter().for_each(
            |ele| *ele=*ele * *ele / f64::from(self.n_samples as f64 - 1.)
            );
        out_ev
    }

    /// getter alias to components_
    pub fn components(&self) -> MatRef<f64>
    {
        self.components_.as_ref()
    }

    /// getter alias to pca_s
    pub fn singular_values(&self) -> MatRef<f64>
    {
        self.pca_s.as_ref()
    }
}


#[cfg(test)]
mod pca_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_pca() {
        // test data (n x p), n n_samples of dim p
        let test_x = random_mat_normal(100, 10);

        // Compute PCA
        let test_pca = PcaRsvd::new(test_x.as_ref(), 4);

        // print results
        let expl_var = test_pca.explained_var();
        let components = test_pca.components();
        print!("explained_var: {:?}", expl_var);
        print!("pca components: {:?}", components);
        print!("pca signular values: {:?}", test_pca.singular_values());
    }
}
