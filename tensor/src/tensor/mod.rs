mod helper;

use std::ops::{Add, Mul, Sub};
use crate::backend::Backend;
use core::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct Tensor<
    T,
    const N: usize,
    const SHAPE: [usize; N], // now allowed on nightly
    B: Backend,
>
where
    [(); N]: ,                // required by generic_const_exprs
{
    _marker: PhantomData<(T, B)>,
}

// impl<T, const N: usize, B: Backend> TensorND<T, N, B> {
//     // todo new function
//
//     pub fn shape(&self) -> &[usize; N] {
//         &self.shape
//     }
// }
//
// // TODO: make these support broadcasting2
// // Multiplication for matrixes of the same number of dims (ex: 2x2 * 2x2)
// impl<T, const N: usize, B: Backend> Mul for TensorND<T, N, B> {
//     type Output = Self;
//
//     fn mul(self, rhs: Self) -> Self::Output {
//         B::mul(self, rhs)
//     }
// }
//
// impl<T, const N: usize, B: Backend> Add for TensorND<T, N, B> {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         B::add(self, rhs)
//     }
// }
//
// impl<T, const N: usize, B: Backend> Sub for TensorND<T, N, B> {
//     type Output = Self;
//
//     fn sub(self, rhs: Self) -> Self::Output {
//         B::sub(self, rhs)
//     }
// }