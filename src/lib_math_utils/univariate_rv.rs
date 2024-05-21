/// Implements univariate random variables
/// including their pdf, and cfd, and how to sample
/// from them.
///
use assert_approx_eq::assert_approx_eq;
use rand_distr::{Normal, Distribution, Beta, Exp};
use rand::Rng;
use rand::seq::index;
use rayon::prelude::*;
use itertools::Itertools;
use ndarray::prelude::*;
use statrs::function::{erf, gamma::{gamma, self}, beta::beta_reg};
use std::f64::consts::PI;
use finitediff::FiniteDiff;
use argmin::{
    core::{CostFunction, Gradient, Error, Executor},
    solver::{particleswarm::ParticleSwarm},
    solver::{linesearch::MoreThuenteLineSearch, linesearch::HagerZhangLineSearch},
    solver::{quasinewton::LBFGS, quasinewton::BFGS, quasinewton::DFP},
    solver::gradientdescent::SteepestDescent,
    solver::neldermead::NelderMead,
};

/// Maximum likelihood fitting
pub fn mlefit(cost: OptMleProblem, method: Option<usize>) -> Result<Vec<f64>, Error> {
    let mut params_opt: Vec<f64> = Vec::new();
    match method.unwrap() {
        // Set up solver
        0 => {
            // SteepestDescent
            let init_param = cost.p_init.clone();
            // set up a line search
            let linesearch = HagerZhangLineSearch::new();
            // Set up solver
            let solver = SteepestDescent::new(linesearch);
            // Run solver
            let res = Executor::new(cost, solver)
                .configure(|state| state.param(init_param).max_iters(40))
                .run()?;
            // extract best results
            let p_best = res.state().clone().take_best_param();
            params_opt = p_best.unwrap();
        }
        1 => {
            // ParticleSwarm
            // annoying conversion from vec to tuple since
            // ParticleSwarm only accepts tuples of
            // vecs for bounds.  TODO: Make upstream PR to argmin crate.
            let ptup_bounds: (_, _) = cost.p_bounds.clone()
                .into_iter()
                .collect_tuple()
                .unwrap();
            let solver = ParticleSwarm::new(ptup_bounds, 40);
            // Run solver
            let res = Executor::new(cost, solver)
                .configure(|state| state.max_iters(100))
                .run()?;
            // extract best results
            let p_best = res.state().clone().take_best_individual();
            params_opt = p_best.unwrap().position;
        }
        2 => {
            // LBFGS
            let init_param = cost.p_init.clone();
            // set up a line search
            let linesearch = HagerZhangLineSearch::new();
            // Set up solver
            let solver = LBFGS::new(linesearch, 4)
                .with_tolerance_grad(1.0e-6)?
                .with_tolerance_cost(1.0e-4)?;
            // Run solver
            let res = Executor::new(cost, solver)
                .configure(|state| state.param(init_param).max_iters(80))
                .run()?;
            // extract best results
            let p_best = res.state().clone().take_best_param();
            params_opt = p_best.unwrap();
        }
        _ => {
           {panic!("Supply valid method: 0=SD, 1=PS, 2:LBFGS");}
        }
    }
    Ok(params_opt)
}


/// Max likelihood fit with particleswarm fallback
pub fn mlefit_ps_fallback(cost: OptMleProblem, method: Option<usize>)
    -> Result<Vec<f64>, Error>
{
    let params_try = mlefit(cost.to_owned(), method);
    match params_try {
        Ok(p) => {
            return Ok(p)
        }
        Err(_perr) => {
            return mlefit(cost.to_owned(), Some(1))
        }
    }
}


/// Optimization problem for maximum likelihood est
#[derive(Clone)]
pub struct OptMleProblem <'a> {
    dist_rv: Box<&'a dyn UniRv>,
    tmp_samples: ArrayView1<'a, f64>,
    p_init: Vec<f64>,
    p_bounds: Vec<Vec<f64>>,
}
impl <'a> OptMleProblem <'a> {
    pub fn new(dist_rv: Box<&'a dyn UniRv>, samples: ArrayView1<'a, f64>, init_params: Vec<f64>, p_bounds: Vec<Vec<f64>>) -> Self
    {
        Self {
            dist_rv: dist_rv,
            tmp_samples: samples,
            p_init: init_params,
            p_bounds,
        }
    }
}
impl <'a> CostFunction for OptMleProblem <'a>  {
    // type aliases
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let fnll = self.dist_rv.nll(self.tmp_samples, Some(p));
        let mut bounds_penalty = 0.0;
        for i in 0..p.len() {
            bounds_penalty += 1.0e1*((p[i] - self.p_bounds[0][i]).min(0.0)).powf(2.0);
            bounds_penalty += 1.0e1*((p[i] - self.p_bounds[1][i]).max(0.0)).powf(2.0);
        }
        Ok(fnll + bounds_penalty)
    }
}
impl <'a> Gradient for OptMleProblem <'a>{
    // type aliases
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*p).forward_diff(&|x| self.cost(x).unwrap()))
//         let fp0 = self.cost(p).unwrap();
//         let eps = 1e-12;
//         let mut out_grad = p.to_owned();
//         for i in 0..out_grad.len() {
//             let mut p_pert = p.to_owned();
//             p_pert[i] += eps;
//             let fpi = self.cost(&p_pert).unwrap();
//             out_grad[i] = (fpi - fp0) / eps
//         }
//         Ok(out_grad)
    }
}


/// Define interface for univariate RVs
// pub trait UniRv<const NPARAM: usize> {
pub trait UniRv: Sync + Send {
    //fn pdff(&self, x: f64, params: Option<[f64; NPARAM]>) -> f64;
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64;
    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64;
    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64>;
    /// Default implementation of negative log likelihood
    fn nll(&self, samples: ArrayView1<f64>, params: Option<&Vec<f64>>) -> f64 {
        // define negative log likelihood to be minimized
        let ln_like: f64 = samples.into_par_iter()
            .map(|x| self.pdf(*x, params).ln()).sum();
        return -ln_like
    }
}


/// Normal distribution parameter storage
#[derive(Debug, Clone)]
pub struct NormalRv  {
    mu: f64,
    std: f64,
    pi_const: f64,
}
impl  NormalRv  {
    pub fn new(mu:f64 , std: f64) -> Self {
        Self {
            mu,
            std,
            pi_const: (2.0 * PI).sqrt(),
        }
    }

    /// Max likelihood est (minimize neg log like) optimization
    pub fn mlfit(&mut self, samples: ArrayView1<f64>, method: Option<usize>)
        -> Result<(), Error>
    {
        // initial guesses for parameters
        let init_params = vec![10., 10.];
        // bounds on parameters
        let p_bounds = vec![
            vec![-1000., 1.0e-12],  // lower param bounds
            vec![1000., 1000.],  // upper param bounds
        ];
        let opt_prob = OptMleProblem::new(
            Box::new(self), samples.view(), init_params, p_bounds);
        let params_opt = mlefit_ps_fallback(opt_prob, method).unwrap();
        self.mu = params_opt[0];
        self.std = params_opt[1];
        Ok(())
    }
}
impl  UniRv for NormalRv  {
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.mu, self.std];
        let par = params.unwrap_or(&tmp_p);
        let mu = par[0];
        let std = par[1].abs();
        (1. / (std * self.pi_const)) *
            (-0.5*((x-mu)/std).powf(2.0)).exp()
    }

    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.mu, self.std];
        let par = params.unwrap_or(&tmp_p);
        let mu = par[0];
        let std = par[1];
        0.5 * (1.0 + erf::erf((x-mu)/(std*(2.0_f64).sqrt())))
    }

    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64> {
        let tmp_p = vec![self.mu, self.std];
        let par = params.unwrap_or(&tmp_p);
        let mu = par[0];
        let std = par[1];
        let rv_f = Normal::new(mu, std).unwrap();
        let mut out = Array1::zeros(n_samples);
        for i in 0..n_samples {
            out[i] = rv_f.sample(&mut rand::thread_rng());
        }
        out
    }
}


/// Beta distribution parameter storage
#[derive(Debug, Clone)]
pub struct BetaRv {
    alpha: f64,
    beta: f64,
    upper_b: f64,
    lower_b: f64,
}
impl BetaRv {
    pub fn new(alpha: f64, beta: f64, lower_b: f64, upper_b: f64) -> Self
    {
        Self {
            alpha,
            beta,
            upper_b,
            lower_b
        }
    }
    /// Max likelihood est (minimize neg log like) optimization
    pub fn mlfit(&mut self, samples: ArrayView1<f64>, method: Option<usize>)
        -> Result<(), Error>
    {
        match method {
            Some(m) => {
                // initial guesses for parameters
                let init_params = vec![1., 1.];
                // bounds on parameters
                let p_bounds = vec![
                    vec![1.0e-4, 1.0e-4],  // lower param bounds
                    vec![200., 200.],  // upper param bounds
                ];
                let opt_prob = OptMleProblem::new(
                    Box::new(self), samples.view(), init_params, p_bounds);
                let params_opt = mlefit_ps_fallback(opt_prob, Some(m)).unwrap();
                self.alpha = params_opt[0];
                self.beta = params_opt[1];
            }
            _ => {
                // direct calculation of alpha, beta parameters from pop samples
                let y_mu = samples.mean().unwrap();
                let y_var = samples.std(1.0).powf(2.0);
                let a = self.lower_b;
                let c = self.upper_b;
                let alpha_d = (a-y_mu)*(a*c-a*y_mu-c*y_mu+y_mu.powf(2.0)+y_var)/
                    (y_var*(c-a));
                let beta_d = -(c-y_mu)*(a*c-a*y_mu-c*y_mu+y_mu.powf(2.0)+y_var)/
                    (y_var*(c-a));
                self.alpha = alpha_d;
                self.beta = beta_d;
            }
        }
        Ok(())
    }
}
impl UniRv for BetaRv {
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        // scale x
        let xs = (x-self.lower_b)/(self.upper_b-self.lower_b);
        let tmp_p = vec![self.alpha, self.beta];
        let par = params.unwrap_or(&tmp_p);
        let alpha = par[0];
        let beta = par[1];
        let b = gamma(alpha)*gamma(beta) / gamma(alpha+beta);
        xs.powf(alpha-1.)*(1.-xs).powf(beta-1.) / b
    }
    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        // scale x
        let xs = (x-self.lower_b)/(self.upper_b-self.lower_b);
        let tmp_p = vec![self.alpha, self.beta];
        let par = params.unwrap_or(&tmp_p);
        let alpha = par[0];
        let beta = par[1];
        beta_reg(alpha, beta, xs)
    }
    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64> {
        let tmp_p = vec![self.alpha, self.beta];
        let par = params.unwrap_or(&tmp_p);
        let alpha = par[0];
        let beta = par[1];
        let rv_f = Beta::new(alpha, beta).unwrap();
        let mut out = Array1::zeros(n_samples);
        for i in 0..n_samples {
            out[i] = (rv_f.sample(&mut rand::thread_rng())*(self.upper_b-self.lower_b))
                     + self.lower_b;
        }
        out
    }
}

/// Exponential distribution parameter storage
#[derive(Debug, Clone)]
pub struct ExponentialRv {
    lambda: f64,
}
impl ExponentialRv {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
    /// Max likelihood est (minimize neg log like) optimization
    pub fn mlfit(&mut self, samples: ArrayView1<f64>, method: Option<usize>)
        -> Result<(), Error>
    {
        // initial guesses for parameters
        let init_params = vec![1.,];
        // bounds on parameters
        let p_bounds = vec![
            vec![1.0e-12,],  // lower param bounds
            vec![100.,],  // upper param bounds
        ];
        let opt_prob = OptMleProblem::new(
            Box::new(self), samples.view(), init_params, p_bounds);
        let params_opt = mlefit_ps_fallback(opt_prob, method).unwrap();
        self.lambda = params_opt[0];
        Ok(())
    }
}
impl UniRv for ExponentialRv {
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.lambda];
        let par = params.unwrap_or(&tmp_p);
        let lambda = par[0];
        lambda*(-lambda*x).exp()
    }
    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.lambda];
        let par = params.unwrap_or(&tmp_p);
        let lambda = par[0];
        1.0-(-lambda*x).exp()
    }
    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64> {
        let tmp_p = vec![self.lambda];
        let par = params.unwrap_or(&tmp_p);
        let lambda = par[0];
        let rv_f = Exp::new(lambda).unwrap();
        let mut out = Array1::zeros(n_samples);
        for i in 0..n_samples {
            out[i] = rv_f.sample(&mut rand::thread_rng());
        }
        out
    }
}

/// Kenerl density estimator parameter storage
#[derive(Debug, Clone)]
pub struct KdeRv {
    bandwidth: f64,
    weights: Array1<f64>,
    kernel: NormalRv,
    supports: Array1<f64>,
}
impl KdeRv {
    pub fn new(init_bandwidth: f64, samples: ArrayView1<f64>) -> Result<Self, Error> {
        let init_self = Self {
            bandwidth: init_bandwidth,
            weights: Array1::ones(samples.len()) / samples.len() as f64,
            kernel: NormalRv::new(0.0, 1.0),
            supports: samples.to_owned(),
        };
        Ok(init_self)
    }
    /// Max likelihood est (minimize neg log like) optimization. The
    /// test samples should be different than support samples in the KDE case!
    /// Split the total sample populatin into a train/test set
    /// use train set for support points of KDE, test set is used for fitting.
    pub fn est_bandwidth(&mut self, test_samples: ArrayView1<f64>, method: Option<usize>)
        -> Result<f64, Error>
    {
        // initial guesses for parameters
        let init_params = vec![self.bandwidth,];
        // bounds on parameters
        let p_bounds = vec![
            vec![1.0e-9,],  // lower param bounds
            vec![1000.,],  // upper param bounds
        ];
        let opt_prob = OptMleProblem::new(
            Box::new(self), test_samples.view(), init_params, p_bounds);
        let params_opt = mlefit_ps_fallback(opt_prob, method).unwrap();
        Ok(params_opt[0])
    }
}
impl UniRv for KdeRv {
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.bandwidth];
        let par = params.unwrap_or(&tmp_p);
        let bandwidth = par[0];
        let mut pdf_f: f64 = 0.0;
        for i in 0..self.weights.len() {
            let sp = self.supports[i];
            pdf_f += self.weights[i] * self.kernel.pdf(x, Some(&vec![sp, bandwidth]));
        }
        pdf_f
    }
    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.bandwidth];
        let par = params.unwrap_or(&tmp_p);
        let bandwidth = par[0];
        let mut cdf_f: f64 = 0.0;
        for i in 0..self.weights.len() {
            let sp = self.supports[i];
            cdf_f += self.weights[i] * self.kernel.cdf(x, Some(&vec![sp, bandwidth]));
        }
        cdf_f
    }
    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64> {
        let tmp_p = vec![self.bandwidth];
        let par = params.unwrap_or(&tmp_p);
        let bandwidth = par[0];
        let mut out = Array1::zeros(n_samples);
        let mut rng = rand::thread_rng();
        for i in 0..n_samples {
            // pick a random kernel
            let rng_idx = rng.gen_range(0..self.weights.len());
            let sp = self.supports[rng_idx];
            let rng_samp = self.kernel.sample(1, Some(&vec![sp, bandwidth]))[0];
            out[i] = rng_samp;
        }
        out
    }
}

/// Construct a KDE from samples automating cross validation method
/// of automatic bandwidth estimation.
pub fn build_kde(init_bandwidth: f64, samples: ArrayView1<f64>, n_iter: usize, method: usize)
    -> Result<KdeRv, Error>
{
    // get many indep estimates of bandwidth
    let mut bandwidth_ests: Vec<f64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _i in 0..n_iter {
        // split the total number of samples into train/test
        let mut support_samples = Vec::new();
        let mut test_samples = Vec::new();
        for s in samples {
            if rng.gen_bool(0.7) {
                support_samples.push(*s);
            }
            else {
                test_samples.push(*s);
            }
        }
        let s_samp = Array1::from_vec(support_samples);
        let s_test = Array1::from_vec(test_samples);
        let bwe = KdeRv::new(init_bandwidth, s_samp.view())
            .unwrap()
            .est_bandwidth(s_test.view(), Some(method));
        bandwidth_ests.push(bwe.unwrap());
    }
    // avg bandwidth
    let bw: f64 = bandwidth_ests.into_iter().sum::<f64>()
        / n_iter as f64;
    KdeRv::new(bw, samples)
}


#[cfg(test)]
mod univariate_rv_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_normal_rv() {
        // generate random samples from known norml dist
        let rv_known = Normal::new(5.25, 10.0).unwrap();
        let ns = 10000;
        let mut tst_s = Array1::zeros(ns);
        for i in 0..ns {tst_s[i] = rv_known.sample(&mut rand::thread_rng()); }

        // init Normal dist with junk initial params
        let mut tst_rv = NormalRv::new(1.0, 1.0);

        // fit test dist to samples by maximum likelihood est
        tst_rv.mlfit(tst_s.view(), Some(1));

        // ensure fitted dist params are correct
        println!("Real pop mean: {:?}, Fitted Mean: {:?}", tst_s.mean(), tst_rv.mu);
        assert_approx_eq!(tst_rv.mu, tst_s.mean().unwrap(), 1e-3);
        assert_approx_eq!(tst_rv.std, tst_s.std(0.0), 1e-3);
    }

    #[test]
    fn test_beta_rv() {
        let test_matrix = vec![
            (1.0, 1.0, 0.0, 1.0),
            (2.0, 2.0, 0.0, 100.0),
            (0.25, 0.75, 2.0, 7.0),
            (1.25, 2.75, 0.2, 0.3),
            (0.25, 2.75, 0.0, 1.0),
            (2.75, 0.25, 0.0, 1.0),
            (0.25, 0.25, 0.0, 1.0),
        ];
        let ns = 40000;
        for (alpha, beta, lower_b, upper_b) in test_matrix.into_iter() {
            let rv_known = Beta::new(alpha, beta).unwrap();
            // draw samples from known dist
            let mut out = Array1::zeros(ns);
            for i in 0..ns {
                out[i] = (rv_known.sample(&mut rand::thread_rng())*(upper_b-lower_b))
                         + lower_b;
            }

            // Init beta dist with junk initial vals
            let mut tst_rv_a = BetaRv::new(1.0, 1.0, lower_b, upper_b);
            // fit to samples
            tst_rv_a.mlfit(out.view(), None);

            // check fitted dist params
            println!("Fitted alpha: {:?}  beta: {:?}", tst_rv_a.alpha, tst_rv_a.beta);
            assert_approx_eq!(tst_rv_a.alpha, alpha, 2e-1);
            assert_approx_eq!(tst_rv_a.beta, beta, 2e-1);

            // sample from fitted beta dist
            let samples = tst_rv_a.sample(ns, None);
            let abs_tol = 7e-2*(upper_b-lower_b);
            assert_approx_eq!(samples.mean().unwrap(), out.mean().unwrap(), abs_tol);
            assert_approx_eq!(samples.std(0.0), out.std(0.0), abs_tol);
        }
    }

    #[test]
    fn test_uniform_rv() {
        // uniform rv is Beta with params (1, 1)
        let tst_rv_a = BetaRv::new(1.0, 1.0, 0.0, 1.0);
        let samples = tst_rv_a.sample(10000, None);
        assert_approx_eq!(samples.mean().unwrap(), 0.5, 1e-2);
    }

    #[test]
    fn test_kde_rv() {
        // generate random samples from known norml dist
        let rv_known = Normal::new(5.25, 10.).unwrap();
        let ns = 100;
        let mut tst_s = Array1::zeros(ns);
        let mut support_s = Array1::zeros(ns);
        for i in 0..ns {tst_s[i] = rv_known.sample(&mut rand::thread_rng()); }
        for i in 0..ns {support_s[i] = rv_known.sample(&mut rand::thread_rng()); }

        // init KDE dist
        let mut kde_dist = KdeRv::new(1.0, support_s.view()).unwrap();
        // estimate optimal bandwidth
        let est_band = kde_dist.est_bandwidth(tst_s.view(), Some(2)).unwrap();
        kde_dist.bandwidth = est_band;

        // sample from fitted KDE dist
        let kde_samples = kde_dist.sample(10000, None);
        println!("Fitted KDE bandwidth: {:?}", kde_dist.bandwidth);
        println!("Real pop mean: {:?}, KDE Mean: {:?}", support_s.mean(), kde_samples.mean());
        println!("Real pop std: {:?}, KDE std: {:?}", support_s.std(0.), kde_samples.std(0.));
        assert_approx_eq!(support_s.mean().unwrap(), kde_samples.mean().unwrap(), 9e-1);
        assert_approx_eq!(support_s.std(0.), kde_samples.std(0.), 3.);

        // test kde automated builder
        let auto_kde_dist = build_kde(1.0, support_s.view(), 10, 2).unwrap();
        let kde_samples = auto_kde_dist.sample(10000, None);
        println!("Fitted KDE bandwidth: {:?}", auto_kde_dist.bandwidth);
        println!("Real pop mean: {:?}, KDE Mean: {:?}", support_s.mean(), kde_samples.mean());
        println!("Real pop std: {:?}, KDE std: {:?}", support_s.std(0.), kde_samples.std(0.));
        assert_approx_eq!(support_s.mean().unwrap(), kde_samples.mean().unwrap(), 9e-1);
        assert_approx_eq!(support_s.std(0.), kde_samples.std(0.), 3.);

    }
}
