/// Integration tests.
/// See individual source files in the src dir for
/// unittest confined to a single module.
use corrla_rs::*;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn it_works() {
        // nullop
    }

    #[test]
    fn test_rbf_rom() {
        // test ability to construct an rbf response surface
        // on low reduced dimensional data.
        // Uses active subspace to obtain most important directions.
        // Projects original data down onto most important directions.
        // Fits a RBF model in the reduced dim space.
    }

    #[test]
    fn test_gp_rom() {
        // test ability to construct an gaussian process response surface
        // on low reduced dimensional data.
        // Uses active subspace to obtain most important directions.
        // Projects original data down onto most important directions.
        // Fits a GP model in the reduced dim space.
    }
}
