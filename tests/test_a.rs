use corrla_rs::*;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = my_add(2, 2);
        assert_eq!(result, 4);
    }
}
