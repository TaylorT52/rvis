#![feature(
    generic_const_exprs,
    generic_const_parameter_types,
    adt_const_params,
    const_trait_impl,
    test
)]
#![feature(portable_simd)]
#![allow(incomplete_features)]

// pub mod conv;
// pub mod pooling;
// pub mod activation;
pub mod relu; 

// pub use conv::*;
// pub use pooling::*;
// pub use activation::*; 
pub use relu::*; 