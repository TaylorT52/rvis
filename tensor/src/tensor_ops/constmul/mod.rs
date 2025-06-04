use crate::storage::HasStorage;
use crate::tensor::Tensor2;
use core::ops::Mul;

pub mod naive_cpu;

/// Trait for backends that support element-wise scalar multiplication.
pub trait ConstMul<T: Copy + Default>: Sized {
    fn constmul<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        k: T,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        T: Mul<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

impl<T, const R: usize, const C: usize, B> Mul<T> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: ConstMul<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::constmul::<R, C>(&self.storage, rhs, &mut out);
        Tensor2 { storage: out, _p: core::marker::PhantomData }
    }
}


