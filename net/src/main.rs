#![feature(generic_const_exprs, generic_const_parameter_types, adt_const_params)]
#![allow(incomplete_features)]

use tensor::storage::naive_cpu::NaiveCpu;
use tensor::tensor::{Tensor2, Tensor3};

fn main() {
    let a = Tensor2::<f32, 3, 3, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = Tensor2::<f32, 3, 3, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let c = (a * b * 5.2) + 1.0;
    let d = a * 5.0;
    let e = d + 1.0;
    let _f = e + 1.0;

    let g = Tensor3::<f32, 2, 3, 3, NaiveCpu>::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    ]);

    let h = Tensor2::<f32, 3, 3, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let i = (g * h * 5.2) + 1.0;

    println!("i =\n{}", i);
}
