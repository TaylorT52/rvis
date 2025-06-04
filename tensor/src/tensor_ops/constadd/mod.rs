//! Element-wise tensor and scalar addition (`+`).
//!
//! Backend implementers should implement [`ConstAdd`] for their backend.
//! Both `Tensor2 + T` and `T + Tensor2` are supported generically for any
//! `T` that implements `Copy + Default + Add`.

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::Tensor2;
use core::ops::Add;

/// Trait for backends that support element-wise scalar addition.
pub trait ConstAdd<T: Copy + Default>: Sized {
    fn constadd<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, { R * C }>;
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
        B::constadd::<R, C>(&self.storage, rhs, &mut out);
        Tensor2 { storage: out, _p: core::marker::PhantomData }
    }
}


