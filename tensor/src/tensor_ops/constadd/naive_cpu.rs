//! NaiveCPU backend – element-wise scalar addition.

use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::constadd::ConstAdd;
use core::ops::Add;

impl<T> ConstAdd<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T>,
{
    fn constadd<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>,
    {
        let src = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { R * C }>>::as_mut_slice(out);

        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s + k;
        }
    }
}
