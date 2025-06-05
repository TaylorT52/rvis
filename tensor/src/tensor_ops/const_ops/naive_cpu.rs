use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::const_ops::{ConstAdd, ConstDiv, ConstMul};
use core::ops::{Add, Div, Mul};

impl<T> ConstAdd<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T>,
{
    fn constadd<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s + k;
        }
    }
}

impl<T> ConstMul<T> for NaiveCpu
where
    T: Copy + Default + Mul<Output = T>,
{
    fn constmul<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s * k;
        }
    }
}

impl<T> ConstDiv<T> for NaiveCpu
where
    T: Copy + Default + Div<Output = T>,
{
    fn constdiv<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s / k;
        }
    }
}
