#![feature(
    generic_const_exprs,
    generic_const_parameter_types,
    adt_const_params,
    const_trait_impl,
    test
)]
#![feature(portable_simd)]
#![allow(incomplete_features)]

extern crate test;

pub mod tensor;
pub mod storage;
pub mod tensor_ops;
