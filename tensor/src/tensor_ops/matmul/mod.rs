pub mod naive_cpu;

use core::ops::{Add, Mul};
use crate::storage::HasStorage;
use crate::tensor::Tensor2;

//trait for backend to implement matmul
pub trait MatMul<T: Copy + Default>: Sized {
    fn matmul<const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * K }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { R * K }>;
}

impl<T, const R: usize, const C: usize, const K: usize, B> Mul<Tensor2<T, C, K, B>>
for Tensor2<T, R, C, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: MatMul<T>
    + HasStorage<T, { R * C }>
    + HasStorage<T, { C * K }>
    + HasStorage<T, { R * K }>,
{
    type Output = Tensor2<T, R, K, B>;

    #[inline]
    fn mul(self, rhs: Tensor2<T, C, K, B>) -> Self::Output {
        // Allocate the output buffer using the backend
        let mut out: <B as HasStorage<T, { R * K }>>::Storage =
            <B as HasStorage<T, { R * K }>>::storage_uninit();

        // Delegate to the backend’s implementation
        B::matmul::<R, C, K>(&self.storage, &rhs.storage, &mut out);

        Self::Output {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
