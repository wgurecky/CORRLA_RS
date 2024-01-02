/// Impl of various sampling methods from
/// constrained and unconstrained problems
/// from various distributions
use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::Dirichlet;

/// Draws samples, x_i such that
///     \sum_i x_i = c_scale
///     and
///     lb_i <= x_i <= ub_i
///
///     # Arguments
///     * `bounds` - 2D ndarray of [[lb_i, ub_i], ...]
///     * `n_samples` - number of samples desired
///     * `max_zshots` - number of shots with each each 'shot' having chunk_size
///     * `chunk_size` - number of samples per shot
///     * `c_scale` - sum constraint
pub fn constr_dirichlet_sample(
    bounds: &Array2<f64>,
    n_samples: usize,
    max_zshots: usize,
    chunk_size: usize,
    c_scale: f64,
    ) -> Array2<f64>
{
    let ndim = bounds.nrows();
    let mut alphas = Vec::new();
    for _i in 0..ndim {
        alphas.push(1.0);
    }

    // storage for final constrained samples
    let mut c_zs: Array2<f64> = Array2::zeros((n_samples, ndim));
    let mut k_valid = 0;
    for _shot in 0..max_zshots {
        // unconstrained samples, uniform in z
        let mut u_zs: Array2<f64> = Array2::zeros((chunk_size, ndim));
        for (_i, mut row) in u_zs.axis_iter_mut(Axis(0)).enumerate() {
            let dirichlet_rv = Dirichlet::new(&alphas).unwrap();
            let u_z_sample = dirichlet_rv.sample(&mut rand::thread_rng());
            row.assign(&Array1::from_vec(u_z_sample));
        }
        u_zs = u_zs * c_scale;

        // Rejection step.
        // Iterate over each sample, check if sample meets
        // all bound constraints. if so, append to output samples
        for (_i, row) in u_zs.axis_iter(Axis(0)).enumerate() {
            let mut row_valid = true;
            for (j, bound) in bounds.axis_iter(Axis(0)).enumerate() {
                let b_valid = (bound[0] <= row[j]) && (row[j] <= bound[1]);
                row_valid = row_valid && b_valid;
            }
            if row_valid {
                c_zs.row_mut(k_valid).assign(&row);
                k_valid += 1;
            }
            if k_valid >= n_samples {
                return c_zs;
            }
        }
    }
    c_zs
}


#[cfg(test)]
mod space_samplers_unit_tests {
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_sampler_constr_dirichlet() {
        let bounds = arr2(&[
             [0.0, 0.0026],
             [0.1955, 0.1995],
             [0.80, 0.825],
             ]);

        let n_samples = 10;
        let samples = constr_dirichlet_sample(&bounds, n_samples, 500, 50000, 1.0);
        print!("Samples: {:?}", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }
    }
}
