pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
use num_traits::Float;

/// Trait for backends that support element-wise logarithm.
pub trait Log<T: Copy + Default>: Sized {
    fn log<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        T: Float,
        Self: HasStorage<T, N>;
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: Copy + Default + Float,
    B: Log<T> + HasStorage<T, { R * C }>,
{
    #[inline]
    pub fn log(self) -> Self {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::log::<{ R * C }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Float,
    B: Log<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn log(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::log::<{ D0 * (D1 * D2) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B> Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Float,
    B: Log<T> + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>,
    [(); D0 * (D1 * (D3 * D4))]:,
{
    #[inline]
    pub fn log(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::log::<{ D0 * (D1 * (D3 * D4)) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
