/// Impl of various sampling methods from
/// constrained and unconstrained problems
/// from various distributions
use ndarray::prelude::*;
use ndarray::{concatenate};
use rand::prelude::*;
use statrs::function::gamma;
use rand::distributions::WeightedIndex;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use assert_approx_eq::assert_approx_eq;

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

struct McmcChain {
    samples: Array2<f64>
}

impl McmcChain {
    pub fn new(ndim: usize, init_s: ArrayView1<f64>) -> Self {
        let mut init_samples: Array2<f64> = Array2::zeros((0, ndim));
        init_samples.push(Axis(0), init_s).unwrap();
        Self {
            samples: init_samples,
        }
    }

    pub fn head_sample(&self) -> ArrayView1<f64> {
        self.samples.row(self.samples.nrows())
    }

    pub fn append_sample(&mut self, sample: ArrayView1<f64>) {
        self.samples.push(Axis(0), sample).unwrap();
    }
}

struct DeMcSampler {
    chains: Vec<McmcChain>,
    chains_id: Vec<usize>,
    gamma: f64,
    n_chains: usize,
    ndim: usize,
    prop_fixup_fn: Option< fn(ArrayView1<f64>) -> Array1<f64> >,
    ln_prob_fn: Box<dyn LnProbFn>,
}

trait LnProbFn {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: Vec<f64>) -> f64;
}

struct LnPriorUniform {
    bounds: Array2<f64>,
}
impl LnPriorUniform {
    pub fn new(bounds: ArrayView2<f64>) -> Self {
        Self {
            bounds: bounds.to_owned(),
        }
    }
}
impl LnProbFn for LnPriorUniform {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: Vec<f64>) -> f64
    {
        let mut valid = false;
        for (i, bound) in self.bounds.axis_iter(Axis(0)).enumerate() {
            valid = valid && bound[0] < sample[i] && sample[i] < bound[1]
        }
        if valid {
            0.0
        }
        else {
            f64::NEG_INFINITY
        }
    }
}

struct LnLikeDirichlet {
    alpha: Array1<f64>,
}
impl LnLikeDirichlet {
    pub fn new(alpha: ArrayView1<f64>) -> Self {
        Self {
            alpha: alpha.to_owned(),
        }
    }
}
impl LnProbFn for LnLikeDirichlet {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: Vec<f64>) -> f64
    {
        // Diriclet PDF is only valid if sum(x)=1
        assert_approx_eq!(sample.sum(), 1.0);
        let alpha_0: f64 = self.alpha.sum();
        let mut numer = 1.0;
        for alph in self.alpha.iter() {
            numer = numer * gamma::gamma(*alph);
        }
        let beta = numer / gamma::gamma(alpha_0);
        let mut pdf = 1.0 / beta;
        for (i, alph) in self.alpha.iter().enumerate() {
            pdf = pdf * sample[i].powf(alph-1.0);
        }
        pdf.ln()
    }
}

impl DeMcSampler {
    pub fn new(ln_prob: impl LnProbFn + 'static, chains: Vec<McmcChain>, ndim: usize, gamma: f64)
        -> Self
    {
        let n_chains = chains.len();
        assert!(n_chains >= 3);
        let mut chains_id = Vec::new();
        for c in 0..n_chains {
            chains_id.push(c.clone());
        }
        Self {
            chains: chains,
            chains_id: chains_id,
            gamma: gamma,
            n_chains: n_chains,
            ndim: ndim,
            prop_fixup_fn: None,
            ln_prob_fn: Box::new(ln_prob),
        }
    }

    pub fn set_prop_fixup(&mut self, fixup_fn: fn(ArrayView1<f64>) -> Array1<f64>) {
        self.prop_fixup_fn = Some(fixup_fn);
    }

    /// draws one sample on specified chain
    fn step_chain(&self, c_idx: usize) -> Array1<f64> {
        let c_sample = self.chains[c_idx].head_sample();
        let mut rng = rand::thread_rng();
        // select two random chains from chain pool that are not the current chain
        let mut possible_chains_id = self.chains_id.clone();
        possible_chains_id.remove(c_idx);
        let sel_ids = possible_chains_id.iter().choose_multiple(&mut rng, 2);
        // compute delta vector
        let a_sample = self.chains[*sel_ids[0]].head_sample();
        let b_sample = self.chains[*sel_ids[1]].head_sample();
        let delta = a_sample.to_owned() - b_sample.to_owned();
        // inject small random noise
        let var_ball: Array1<f64> = Array1::zeros(self.ndim);
        // construct proposal
        let mut prop_sample = c_sample.to_owned() + self.gamma * delta + var_ball;
        // optionally fix-up sample
        match self.prop_fixup_fn {
            Some(_fn) => { prop_sample = _fn(prop_sample.view()); }
            None => { }
        }
        // compute prob ratio
        let alpha = self.accept_prob_ratio(c_sample.view(), prop_sample.view());
        let accept = self.metropolis_accept(alpha);
        if accept {
            prop_sample
        }
        else {
            c_sample.to_owned()
        }
    }

    /// draws n_samples on all chains
    pub fn sample_mcmc(&mut self, n_samples: usize) {
        for _s in 0..n_samples {
            for c_i in 0..self.chains_id.len() {
                let c_sample = self.step_chain(self.chains_id[c_i]);
                self.chains[c_i].append_sample(c_sample.view());
            }
        }
    }

    fn accept_prob_ratio(&self, cur_sample: ArrayView1<f64>, prop_sample: ArrayView1<f64>) -> f64
    {
        let lnp_diff = self.ln_prob_fn.lnp(prop_sample, Vec::new()) -
                       self.ln_prob_fn.lnp(cur_sample, Vec::new());
        let mut alpha: f64 = lnp_diff.max(1.0);
        alpha = alpha.max(0.0);
        alpha = alpha.min(1.0);
        alpha
    }

    fn metropolis_accept(&self, alpha: f64) -> bool {
        let mut rng = rand::thread_rng();
        let weights = [alpha, 1. - alpha];
        // flip unfair coin
        let dist = WeightedIndex::new(&weights).unwrap();
        let choices = [true, false];
        choices[dist.sample(&mut rng)]
    }
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
