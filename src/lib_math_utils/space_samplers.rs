/// Impl of various sampling methods from
/// constrained and unconstrained problems
/// from various distributions
use ndarray::prelude::*;
use ndarray::{concatenate};
use std::{thread, time::Duration};
use rand::prelude::*;
use statrs::function::gamma;
use rand::distributions::{WeightedIndex, Uniform};
use rand_distr::{Dirichlet, Normal};
use rayon::prelude::*;
use assert_approx_eq::assert_approx_eq;

fn dirichlet_shot_sample(
    bounds: ArrayView2<f64>,
    mut c_zs: ArrayViewMut2<f64>,
    max_zshots: usize,
    chunk_size: usize,
    c_scale: f64,
    alphas: &Vec<f64>
    )
{
    let ndim = bounds.nrows();
    let n_samples = c_zs.nrows();
    let mut k_valid = 0;
    for _shot in 0..max_zshots {
        // unconstrained samples, uniform in z
        let mut u_zs: Array2<f64> = Array2::zeros((chunk_size, ndim));
        for (_i, mut row) in u_zs.axis_iter_mut(Axis(0)).enumerate() {
            let dirichlet_rv = Dirichlet::new(alphas).unwrap();
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
    in_alphas: Option<Vec<f64>>
    ) -> Array2<f64>
{
    let ndim = bounds.nrows();
    // set default Diriclet dist shape params if not specified
    let mut alphas = Vec::new();
    match in_alphas {
        Some(in_alphas) => {
            if in_alphas.len() == ndim {
                alphas = in_alphas;
            }
            else if in_alphas.len() == 1 {
                for _i in 0..ndim {
                    alphas.push(in_alphas[0]);
                }
            }
            else {
                panic!("Number of shape parameters to Diriclet sampler must be ndim or 1 for the sym case");
            }
        }
        _ => {
            for _i in 0..ndim {
                alphas.push(1.0);
            }
        }
    }

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
                    chunk_size, c_scale, &alphas);
        });
    // collect samples
    let mut all_zshot_view = Vec::new();
    for p in 0..all_zshots.len() {
        all_zshot_view.push(all_zshots[p].view());
    }
    let all_samples = concatenate(Axis(0), &all_zshot_view).unwrap();
    all_samples
}

pub struct McmcChain {
    samples: Array2<f64>,
    id: usize,
}

impl McmcChain {
    pub fn new(ndim: usize, init_s: ArrayView1<f64>, id: usize) -> Self {
        let mut init_samples: Array2<f64> = Array2::zeros((0, ndim));
        init_samples.push(Axis(0), init_s).unwrap();
        Self {
            samples: init_samples,
            id: id,
        }
    }

    pub fn head_sample(&self) -> ArrayView1<f64> {
        self.samples.row(self.samples.nrows()-1)
    }

    pub fn append_sample(&mut self, sample: ArrayView1<f64>) {
        self.samples.push(Axis(0), sample).unwrap();
    }
}


/// Interface for log probability methods used in MCMC samplers
trait LnProbFn: Sync + Send {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: &Vec<f64>) -> f64;
}

/// A custom likelihood function
struct LnLikeCustom {
    ln_like_fn: Box<dyn Fn(ArrayView1<f64>, &Vec<f64>)->f64 + Send + Sync>,
}
impl LnLikeCustom {
    pub fn new(lnf: impl Fn(ArrayView1<f64>, &Vec<f64>)->f64 + 'static + Send + Sync) -> Self {
        Self { ln_like_fn: Box::new(lnf) }
    }
}
impl LnProbFn for LnLikeCustom {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: &Vec<f64>) -> f64
    {
        (self.ln_like_fn)(sample, extra_args)
    }
}

/// A flat prior
pub struct LnPriorUniform {
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
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: &Vec<f64>) -> f64
    {
        let mut valid = true;
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

/// A Dirichlet MV likelihood
pub struct LnLikeDirichlet {
    alpha: Vec<f64>,
}
impl LnLikeDirichlet {
    pub fn new(alpha: &Vec<f64>) -> Self {
        Self {
            alpha: alpha.to_owned(),
        }
    }
}
impl LnProbFn for LnLikeDirichlet {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: &Vec<f64>) -> f64
    {
        // Diriclet PDF is only valid if sum(x)=1
        assert_approx_eq!(sample.sum(), 1.0);
        let alpha_0: f64 = self.alpha.iter().sum();
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

/// Compose the numerator of the Posterior PDF (prior * likelihood)
pub struct LnLikeSum {
    like_fn: Box<dyn LnProbFn>,
    prior_fn: Box<dyn LnProbFn>,
}
impl LnLikeSum {
    pub fn new(like: impl LnProbFn + 'static, prior: impl LnProbFn + 'static) -> Self
    {
        Self {
            like_fn: Box::new(like),
            prior_fn: Box::new(prior),
        }
    }
}
impl LnProbFn for LnLikeSum {
    fn lnp(&self, sample: ArrayView1<f64>, extra_args: &Vec<f64>) -> f64
    {
        self.like_fn.lnp(sample, extra_args) + self.prior_fn.lnp(sample, extra_args)
    }
}

pub struct DeMcSampler {
    chains: Vec<McmcChain>,
    chains_id: Vec<usize>,
    gamma: f64,
    n_chains: usize,
    ndim: usize,
    var_epsilon: f64,
    prop_fixup_fn: Option< Box<dyn Fn(ArrayView1<f64>)->Array1<f64> + Send + Sync> >,
    ln_prob_fn: Box<dyn LnProbFn>,
    n_accept: usize,
    n_reject: usize,
}

impl DeMcSampler {
    pub fn new(ln_prob: impl LnProbFn + 'static, chains: Vec<McmcChain>, ndim: usize, gamma: f64, var_eps: f64)
        -> Self
    {
        let n_chains = chains.len();
        assert!(n_chains >= 3);
        let mut chains_id = Vec::new();
        for c in 0..n_chains {
            assert!(chains[c].samples.ncols() == ndim);
            chains_id.push(c.clone());
        }
        Self {
            chains: chains,
            chains_id: chains_id,
            gamma: gamma,
            n_chains: n_chains,
            ndim: ndim,
            var_epsilon: var_eps,
            prop_fixup_fn: None,
            ln_prob_fn: Box::new(ln_prob),
            n_accept: 0,
            n_reject: 0,
        }
    }

    pub fn set_prop_fixup(&mut self, fixup_fn: impl Fn(ArrayView1<f64>)->Array1<f64> + 'static + Sync + Send) {
        self.prop_fixup_fn = Some(Box::new(fixup_fn));
    }

    pub fn n_samples(&self) -> usize {
        self.chains[0].samples.len()
    }

    /// Get the last n_tail samples from the given chain id
    /// If n_tail = 0, returns all samples
    pub fn get_chain_samples(&self, n_tail: usize, id: usize) -> Array2<f64> {
        // self.chains[id].samples.ro
        // self.chains[id].samples.view()
        let n_s: i32 = n_tail as i32;
        self.chains[id].samples.slice(s![-n_s.., ..]).to_owned()
    }

    /// Get the last n_tail samples from all chains
    /// If n_tail = 0, returns all samples
    pub fn get_samples(&self, n_tail: usize) -> Array2<f64> {
        let mut tmp_out: Vec<Array2<f64>> = Vec::new();
        for c in 0..self.n_chains {
            tmp_out.push(self.get_chain_samples(n_tail, c));
        }
        let mut out: Array2<f64> = Array2::zeros((n_tail*self.n_chains, self.ndim));
        let mut i: usize = 0;
        for s in 0..n_tail {
            for c in 0..self.n_chains {
                out.row_mut(i).assign(&tmp_out[c].row(s));
                i += 1;
            }
        }
        out
    }

    /// draws one sample on specified chain
    fn step_chain(&self, c_idx: usize) -> (bool, Array1<f64>) {
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
        let u_dist = Uniform::new(0.0, self.var_epsilon);
        let uni_samples = (0..self.ndim).into_iter().map(|_i| { u_dist.sample(&mut rng) } ).collect();
        let var_ball = Array1::from_vec(uni_samples);

        // construct proposal
        let mut prop_sample = c_sample.to_owned() + self.gamma * delta + var_ball;
        // optionally fix-up sample
        match &self.prop_fixup_fn {
            Some(_fn) => { prop_sample = _fn(prop_sample.view()); }
            None => { }
        }
        // compute prob ratio
        let alpha = self.accept_prob_ratio(c_sample.view(), prop_sample.view());
        let accept = self.metropolis_accept(alpha);
        if accept {
            (true, prop_sample)
        }
        else {
            (false, c_sample.to_owned())
        }
    }

    /// draws n_samples on all chains
    pub fn sample_mcmc(&mut self, n_samples: usize) {
        for _s in 0..n_samples {
            for c_i in 0..self.chains_id.len() {
                let (act, c_sample) = self.step_chain(self.chains_id[c_i]);
                if act {
                    self.n_accept += 1;
                }
                else {
                    self.n_reject += 1;
                }
                self.chains[c_i].append_sample(c_sample.view());
            }
        }
    }

    /// draw n_samples on all chains in parallel
    pub fn sample_mcmc_par(&mut self, n_samples: usize) {
        for _s in 0..n_samples {
            let mut p_sample_vec = Vec::new();
            p_sample_vec = (0..self.n_chains).into_par_iter()
                .map(|c_i| { self.step_chain(self.chains_id[c_i]) })
                .collect();
            for (c_i, (act, p_samp)) in p_sample_vec.iter().enumerate() {
                if *act {
                    self.n_accept += 1;
                }
                else {
                    self.n_reject += 1;
                }
                self.chains[c_i].append_sample(p_samp.view());
            }
        }
    }

    /// Global acceptance ratio
    pub fn accept_ratio(&self) -> f64 {
        self.n_accept as f64 / (self.n_reject as f64 + self.n_accept as f64)
    }

    fn accept_prob_ratio(&self, cur_sample: ArrayView1<f64>, prop_sample: ArrayView1<f64>) -> f64
    {
        let prop_lnp = self.ln_prob_fn.lnp(prop_sample, &Vec::new());
        let cur_lnp = self.ln_prob_fn.lnp(cur_sample, &Vec::new());
        let mut alpha = (prop_lnp - cur_lnp).exp();
        alpha = alpha.min(1.0);
        alpha = alpha.max(0.0);
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
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0, None);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }

        let n_samples = 13;
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0, None);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }

        let n_samples = 21;
        let samples = constr_dirichlet_sample(bounds.view(), n_samples, 500, 20000, 1.0, None);
        print!("Samples: {:?} \n", samples);
        assert_eq!(samples.nrows(), n_samples);
        for (_i, sample) in samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }
    }

    #[test]
    fn test_demcmc_sampler_gauss() {
        // Sample 1D Gaussian distribution using the mcmc sampler.
        let tst_mu = 2.0;
        let tst_std = 3.0;
        // log likelihood up to a constant
        let tst_gauss_pdf = move | x: ArrayView1<f64>, _b: &Vec<f64> | { (-0.5*((x[0]-tst_mu)/tst_std).powf(2.0)).exp()/tst_std } ;
        let tst_gauss_ln_pdf = move | x: ArrayView1<f64>, _b: &Vec<f64> | { tst_gauss_pdf(x, _b).ln() } ;
        // construct bounds prior
        let bounds = Array2::from(vec![
            [-20.0, 20.0]
            ]);
        let tst_ln_prior = LnPriorUniform::new(bounds.view());
        // construct likelihood
        let tst_ln_like = LnLikeCustom::new(tst_gauss_ln_pdf);
        // construct likelihood*prior
        let tst_ln_like_prior = LnLikeSum::new(tst_ln_like, tst_ln_prior);

        // init the MCMC sampler chains
        let x0 = Array1::from(vec![0.0]);
        let mut tst_chains: Vec<McmcChain> = Vec::new();
        let n_chains = 8;
        for c in 0..n_chains {
            tst_chains.push(McmcChain::new(1, x0.view(), c));
        }
        // init the MCMC sampler
        let mut tst_mcmc_sampler = DeMcSampler::new(tst_ln_like_prior, tst_chains, 1, 0.8, 1.0e-10);

        // draw samples
        let n_samples: usize = 5000;
        tst_mcmc_sampler.sample_mcmc(n_samples);

        // Check the estimated mean and var against known]
        let tst_samples = tst_mcmc_sampler.get_samples(2000);
        let ar = tst_mcmc_sampler.accept_ratio();
        println!("MCMC Samples: {:?}", tst_samples);
        println!("Accept ratio: {:?}", ar);

        // compute sample mean
        let tst_sample_std = tst_samples.std(1.0);
        let tst_sample_mu = tst_samples.mean().unwrap();
        println!("Mean, std: {:?}, {:?}", tst_sample_mu, tst_sample_std);
        assert_approx_eq!(tst_sample_mu, tst_mu, 5.0e-1);
        assert_approx_eq!(tst_sample_std, tst_std, 5.0e-1);
        assert!(ar > 0.2);
    }

    #[test]
    fn test_demcmc_dirichlet() {
        // TODO: test ability to draw samples from
        // constrained 3D dirichlet dist.
        // This is useful to greatly improve efficiency of drawing
        // samples froma highly constrained dirichlet sampling problem
        // in high dimensions where rejection sampling alone is inefficient.
        let bounds = arr2(&[
             [0.0, 0.0026],
             [0.1955, 0.1995],
             [0.80, 0.825],
             ]);

        // Draw a few samples that fall within the feasible region using rejection sampling
        let n_seed_samples = 8;
        let diri_samples = constr_dirichlet_sample(bounds.view(), n_seed_samples, 500, 20000, 1.0, None);

        // Bounds for MCMC prior
        let tst_ln_prior = LnPriorUniform::new(bounds.view());
        // construct likelihood
        let alphas = vec![1.0, 1.0, 1.0];
        let tst_ln_like = LnLikeDirichlet::new(&alphas);
        // construct likelihood*prior
        let tst_ln_like_prior = LnLikeSum::new(tst_ln_like, tst_ln_prior);

        // define fixup function
        // this is required for sampling from the MV Diriclet dist since
        // the support is only on the simplex sum_i x_i = 1, and the
        // mcmc proposal x_i may fall off this simplex
        let proposal_fix_fn = | x: ArrayView1<f64> | -> Array1<f64> {
            let mut new_x = x.to_owned();
            // must sum to 1
            thread::sleep(Duration::from_millis(1));
            new_x = new_x.clone() / new_x.sum();
            new_x
        };

        // initialize chains at seed locations
        let mut tst_chains: Vec<McmcChain> = Vec::new();
        for (c, seed_s) in diri_samples.rows().into_iter().enumerate() {
            tst_chains.push(McmcChain::new(3, seed_s, c));
        }
        // init the MCMC sampler
        let mut tst_mcmc_sampler = DeMcSampler::new(tst_ln_like_prior, tst_chains, 3, 0.8, 1.0e-10);

        // specify the proposal fixup function
        tst_mcmc_sampler.set_prop_fixup(proposal_fix_fn);

        // draw samples
        let n_samples: usize = 5000;
        tst_mcmc_sampler.sample_mcmc_par(n_samples);
        let ar = tst_mcmc_sampler.accept_ratio();
        println!("Accept ratio: {:?}", ar);

        // TODO: check samples
        let tst_samples = tst_mcmc_sampler.get_samples(2000/8);
        println!("MCMC Diriclet Samples: {:?}", tst_samples);
        // assert_eq!(tst_samples.nrows(), n_samples);
        for (_i, sample) in tst_samples.axis_iter(Axis(0)).enumerate() {
            assert_approx_eq!(sample.sum(), 1.0);
        }
    }
}
