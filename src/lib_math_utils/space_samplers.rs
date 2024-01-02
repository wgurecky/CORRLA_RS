/// Impl of various sampling methods from
/// constrained and unconstrained problems
/// from various distributions
use ndarray::prelude::*;
use ndarray::{concatenate};
use rand::prelude::*;
use rand_distr::Dirichlet;
use rayon::prelude::*;

fn dirichlet_shot_sample(
    bounds: ArrayView2<f64>,
    mut c_zs: ArrayViewMut2<f64>,
    max_zshots: usize,
    chunk_size: usize,
    c_scale: f64
    )
{
    let mut alphas = Vec::new();
    let ndim = bounds.nrows();
    for _i in 0..ndim {
        alphas.push(1.0);
    }
    let n_samples = c_zs.nrows();
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
                break;
            }
        }
        if k_valid >= n_samples {
            break;
        }
    }
}

/// Draws samples, x_i such that
///     \sum_i x_i = c_scale
///     and
///     lb_i <= x_i <= ub_i
///
pub fn constr_dirichlet_sample(
    bounds: ArrayView2<f64>,
    n_samples: usize,
    max_zshots: usize,
    chunk_size: usize,
    c_scale: f64,
    ) -> Array2<f64>
{
    let ndim = bounds.nrows();
    // split total number of desired samples into chunks
    let n_par = std::cmp::min(n_samples, 10);
    let mut all_zshots = Vec::new();
    let mut avail_samples = n_samples;
    let mut local_n_samples = n_samples / n_par;
    loop {
        if local_n_samples > avail_samples {
            local_n_samples = avail_samples;
        }
        let mut zs: Array2<f64> = Array2::zeros((local_n_samples, ndim));
        all_zshots.push(zs);
        avail_samples = std::cmp::max(avail_samples-local_n_samples, 0);
        if avail_samples <= 0 {
            break;
        }
    }
    // parallel sample
    all_zshots.par_iter_mut().for_each(|c_zs|
        {
            dirichlet_shot_sample(bounds, c_zs.view_mut(), max_zshots,
                    chunk_size, c_scale);
        });
    // collect samples
    let mut all_zshot_view = Vec::new();
    for p in 0..all_zshots.len() {
        all_zshot_view.push(all_zshots[p].view());
    }
    let all_samples = concatenate(Axis(0), &all_zshot_view).unwrap();
    all_samples
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

        let n_samples = 8;
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }

        let n_samples = 13;
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }

        let n_samples = 21;
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }
    }
}
