//! Element-wise tensor and scalar addition (`+`).
//!
//! Backend implementers should implement [`ConstAdd`] for their backend.
//! Both `Tensor2 + T` and `T + Tensor2` are supported generically for any
//! `T` that implements `Copy + Default + Add`.

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
use core::ops::Add;

/// Trait for backends that support element-wise scalar addition.
pub trait ConstAdd<T: Copy + Default>: Sized {
    fn constadd<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, N>;
}

impl<T, const R: usize, const C: usize, B> Add<T> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Add<Output = T>,
    B: ConstAdd<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::constadd::<{ R * C }>(&self.storage, rhs, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Add<T> for Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Add<Output = T>,
    B: ConstAdd<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    type Output = Tensor3<T, D0, D1, D2, B>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::constadd::<{ D0 * (D1 * D2) }>(&self.storage, rhs, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B> Add<T>
    for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Add<Output = T>,
    B: ConstAdd<T> + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>,
    [(); D0 * (D1 * (D3 * D4))]:,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::constadd::<{ D0 * (D1 * (D3 * D4)) }>(&self.storage, rhs, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
