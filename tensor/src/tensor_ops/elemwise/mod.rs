//! Element-wise tensor operations between two tensors of the same size.
//!
//! Backend implementers should implement [`ElemAdd`], [`ElemSub`], and
//! [`ElemMul`] for their backend.

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::Tensor2;
use core::ops::{Add, Mul, Sub};

/// Trait for backends that support element-wise tensor addition.
pub trait ElemAdd<T: Copy + Default>: Sized {
    fn elem_add<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

/// Trait for backends that support element-wise tensor subtraction.
pub trait ElemSub<T: Copy + Default>: Sized {
    fn elem_sub<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        T: Sub<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

/// Trait for backends that support element-wise tensor multiplication.
pub trait ElemMul<T: Copy + Default>: Sized {
    fn elem_mul<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        T: Mul<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

impl<T, const R: usize, const C: usize, B> Add<Tensor2<T, R, C, B>> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Add<Output = T>,
    B: ElemAdd<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn add(self, rhs: Tensor2<T, R, C, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::elem_add::<R, C>(&self.storage, &rhs.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize, B> Sub<Tensor2<T, R, C, B>> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Sub<Output = T>,
    B: ElemSub<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn sub(self, rhs: Tensor2<T, R, C, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::elem_sub::<R, C>(&self.storage, &rhs.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: ElemMul<T> + HasStorage<T, { R * C }>,
{
    #[inline]
    pub fn elem_mul(self, rhs: Tensor2<T, R, C, B>) -> Self {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::elem_mul::<R, C>(&self.storage, &rhs.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
