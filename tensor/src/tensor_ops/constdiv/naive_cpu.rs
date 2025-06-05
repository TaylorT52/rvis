use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::constdiv::ConstDiv;
use core::ops::Div;

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
