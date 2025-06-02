use crate::backend::Backend;

use core::ops::{Add, Div, Mul, Sub};

pub struct NaiveCpu;

impl<T> Backend<T> for NaiveCpu
where
    T: Copy
    + Default
    + Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div<Output = T>,
{
    type Storage = Vec<T>;

    #[inline]
    fn storage_from_vec(src: Vec<T>) -> Self::Storage { src }

    #[inline]
    fn storage_uninit(len: usize) -> Self::Storage { vec![T::default(); len] }

    #[inline]
    fn as_slice(s: &Self::Storage) -> &[T] { s }

    #[inline]
    fn as_mut_slice(s: &mut Self::Storage) -> &mut [T] { s }

    #[inline]
    fn add(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage) {
        for ((o, a), b) in out.iter_mut().zip(lhs).zip(rhs) { *o = *a + *b; }
    }
    #[inline]
    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage) {
        for ((o, a), b) in out.iter_mut().zip(lhs).zip(rhs) { *o = *a - *b; }
    }
    #[inline]
    fn mul(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage) {
        for ((o, a), b) in out.iter_mut().zip(lhs).zip(rhs) { *o = *a * *b; }
    }
    #[inline]
    fn div(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage) {
        for ((o, a), b) in out.iter_mut().zip(lhs).zip(rhs) { *o = *a / *b; }
    }

    #[inline]
    fn add_scalar(lhs: &Self::Storage, rhs: T, out: &mut Self::Storage) {
        for (o, a) in out.iter_mut().zip(lhs) { *o = *a + rhs; }
    }
    #[inline]
    fn mul_scalar(lhs: &Self::Storage, rhs: T, out: &mut Self::Storage) {
        for (o, a) in out.iter_mut().zip(lhs) { *o = *a * rhs; }
    }
}