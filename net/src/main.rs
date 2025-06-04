#![feature(generic_const_exprs, generic_const_parameter_types, adt_const_params)]
#![allow(incomplete_features)]

use tensor::storage::naive_cpu::NaiveCpu;
use tensor::tensor::{StaticShape, Tensor2};

fn main() {
    let a = Tensor2::<f32, 2, 3, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor2::<f32, 3, 2, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = b * 5;
    let d = 5 * a;
    let e = d + 1;
    let f = 1 + e;
    println!("{:?}", c.shape());
}
