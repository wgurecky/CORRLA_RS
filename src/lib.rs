// root level lib
use faer::{prelude::*};
use faer_core::{mat, Mat, MatRef};

// other modules
pub mod lib_math_utils;
pub mod lib_math_utils_py;

// example root level routines in the lib
pub fn my_add(left: usize, right: usize) -> usize {
    left + right
}


pub fn svd_random_faer<'a>(my_mat: Mat<f64>) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    print!("test");
    print!("test2");
    let my_svd = my_mat.svd();
    (my_svd.u().to_owned(), my_svd.s_diagonal().to_owned(), my_svd.v().to_owned())
}


// example simple within-lib test
#[cfg(test)]
mod lib_unit_tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = my_add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn tst_svd(){
        let a = mat![
            [1., 2.],
            [3., 4.],
            ];
        // multiple assignment with types
        let (b, c, d): (Mat<f64>,Mat<f64>,Mat<f64>) = svd_random_faer(a);
        print!("svd test \n");
        print!("{:?} \n", b);
    }
}
