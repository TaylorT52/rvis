use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::exp::{Exp, ExpElem};

impl<T> Exp<T> for NaiveCpu
where
    T: ExpElem,
{
    fn exp<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        for i in 0..N {
            dst[i] = src[i].exp();
        }
    }
}
