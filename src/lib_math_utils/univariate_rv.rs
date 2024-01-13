/// Implements univariate random variables
/// including their pdf, and cfd, and how to sample
/// from them.
///
use rand_distr::{Normal, Distribution};
use ndarray::prelude::*;
use statrs::function::erf;
use std::f64::consts::PI;
use finitediff::FiniteDiff;
use argmin::{
    core::{CostFunction, Gradient, Error, Executor},
    solver::{particleswarm::ParticleSwarm},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};


/// Define interface for univariate RVs
pub trait UniRv<const NPARAM: usize> {
    //fn pdff(&self, x: f64, params: Option<[f64; NPARAM]>) -> f64;
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64;
    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64;
    fn sample(&self, n_samples: usize, params: Option<&Vec<f64>>) -> Array1<f64>;
    /// Default implementation of negative log likelihood
    fn nll(&self, samples: ArrayView1<f64>, params: Option<&Vec<f64>>) -> f64 {
        let mut ln_like: f64 = 0.0;
        // define negative log likelihood to be minimized
        for x in samples {
            ln_like += self.pdf(*x, params).ln();
        }
        return -ln_like
    }
}

/// Normal distribution parameter storage
pub struct NormalRv  {
    mu: f64,
    std: f64,
    pi_const: f64,
    tmp_samples: Option<Array1<f64>>,
}
impl  CostFunction for NormalRv   {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let fnll = self.nll(self.tmp_samples.as_ref().unwrap().view(), Some(p));
        println!("Params: {:?}, nll: {:?}", Some(p), fnll);
        Ok(fnll)
    }
}
impl Gradient for NormalRv {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//         Ok((*p).forward_diff(&|x|
//                 self.nll(self.tmp_samples.as_ref().unwrap().view(), Some(x))
//                 )
//           )
        let fp0 = self.nll(self.tmp_samples.as_ref().unwrap().view(), Some(p));
        let eps = 1e-12;
        let mut out_grad = p.to_owned();
        for i in 0..out_grad.len() {
            let mut p_pert = p.to_owned();
            p_pert[i] += eps;
            let fpi = self.nll(self.tmp_samples.as_ref().unwrap().view(), Some(&p_pert));
            out_grad[i] = (fpi - fp0) / eps
        }
        Ok(out_grad)
    }
}
impl  NormalRv  {
    pub fn new(mu:f64 , std: f64) -> Self {
        Self {
            mu,
            std,
            pi_const: (2.0 * PI).sqrt(),
            tmp_samples: None,
        }
    }

    /// Max likelihood est (minimize neg log like) optimization
    fn mlfit(&mut self, samples: ArrayView1<f64>, method: Option<usize>) -> Result<(), Error> {
        let cost = Self{
            mu: 0.0,
            std: 1.0,
            pi_const: self.pi_const,
            tmp_samples: Some(samples.to_owned()),
        };

        match method.unwrap() {
            1 => {
                // Set up solver
                let p_bounds = (vec![-1000.0, 1.0e-12], vec![1000.0, 1000.0]);
                let solver = ParticleSwarm::new(p_bounds, 40);
                // Run solver
                let res = Executor::new(cost, solver)
                    .configure(|state| state.max_iters(100))
                    .run()?;
                // extract best results
                let p_best = res.state().clone().take_best_individual();
                match p_best {
                    Some(p_best) => {
                        let params_opt = p_best.position;
                        self.mu = params_opt[0];
                        self.std = params_opt[1];
                    }
                    _ => {println!("Fitting Failed");}
                }
            }
            2 => {
                let init_param = vec![10.0, 10.0];
                // set up a line search
                let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
                // Set up solver
                let solver = LBFGS::new(linesearch, 7);
                // Run solver
                let res = Executor::new(cost, solver)
                    .configure(|state| state.param(init_param).max_iters(200))
                    .run()?;
                // extract best results
                let p_best = res.state().clone().take_best_param();
                match p_best {
                    Some(p_best) => {
                        let params_opt = p_best;
                        self.mu = params_opt[0];
                        self.std = params_opt[1];
                    }
                    _ => {println!("Fitting Failed");}
                }

            }
            _ => {
               {println!("Supply valid method");}
            }
        }
        Ok(())
    }
}
impl  UniRv<2> for NormalRv  {
    fn pdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.mu, self.std];
        let par = params.unwrap_or(&tmp_p);
        let mu = par[0];
        let std = par[1].abs();
        1. / (std * self.pi_const) *
            (-0.5*((x-mu)/std).powf(2.0)).exp()
    }

    fn cdf(&self, x: f64, params: Option<&Vec<f64>>) -> f64 {
        let tmp_p = vec![self.mu, self.std];
        let par = params.unwrap_or(&tmp_p);
        let mu = par[0];
        let std = par[1];
        0.5 *
        (1.0 + erf::erf((x-mu)/(std*(2.0_f64).sqrt())))
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
pub struct BetaRv {
    alpha: f64,
    beta: f64,
    upper_b: f64,
    lower_b: f64,
}


/// Exponential distribution parameter storage
pub struct ExponentialRv {
    lambda: f64,
}


pub struct KdeRv {
    bandwidth: f64,
    weights: Vec<f64>,
    // kernels: Vec<NormalRv>,
}


#[cfg(test)]
mod univariate_rv_unit_tests {
    // bring everything from above (parent) module into scope
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_normal_rv() {
        // generate random samples from known norml dist
        let rv_known = Normal::new(5.25, 10.0).unwrap();
        let ns = 1000;
        let mut tst_s = Array1::zeros(ns);
        for i in 0..ns {tst_s[i] = rv_known.sample(&mut rand::thread_rng()); }

        // init Normal dist with junk initial params
        let mut tst_rv = NormalRv::new(1.0, 1.0);

        // fit test dist to samples by maximum likelihood est
        tst_rv.mlfit(tst_s.view(), Some(1));

        // ensure fitted dist params are correct
        println!("Real pop mean: {:?}", tst_s.mean());
        assert_approx_eq!(tst_rv.mu, tst_s.mean().unwrap(), 1e-4);
        assert_approx_eq!(tst_rv.std, tst_s.std(0.0), 1e-4);
    }
}
