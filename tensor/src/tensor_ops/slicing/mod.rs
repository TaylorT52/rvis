//! Tensor slicing operations.
//!
//! Backend implementers should implement [`Slice`] for their backend.

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};

/// Trait for backends that support tensor slicing.
pub trait Slice<T: Copy + Default>: Sized {
    /// Slice a 2D tensor along both dimensions
    fn slice2<const R: usize, const C: usize, const R_START: usize, const R_END: usize, const C_START: usize, const C_END: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }> + HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>;

    /// Slice a 3D tensor along any dimension
    fn slice3<const D0: usize, const D1: usize, const D2: usize, const DIM: usize, const START: usize, const END: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }>;

    /// Slice a 4D tensor along any dimension
    fn slice4<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const DIM: usize, const START: usize, const END: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>;
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: Copy + Default,
    B: Slice<T> + HasStorage<T, { R * C }>,
{
    /// Slice the tensor along both dimensions
    pub fn slice<const R_START: usize, const R_END: usize, const C_START: usize, const C_END: usize>(
        &self,
    ) -> Tensor2<T, { R_END - R_START }, { C_END - C_START }, B>
    where
        B: HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>,
    {
        let mut out = <B as HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>>::storage_uninit();
        B::slice2::<R, C, R_START, R_END, C_START, C_END>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default,
    B: Slice<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    /// Slice the tensor along the specified dimension
    pub fn slice<const DIM: usize, const START: usize, const END: usize>(&self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::slice3::<D0, D1, D2, DIM, START, END>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, B>
    Tensor4<T, D0, D1, D2, D3, B>
where
    T: Copy + Default,
    B: Slice<T> + HasStorage<T, { D0 * (D1 * (D2 * D3)) }>,
    [(); D0 * (D1 * (D2 * D3))]:,
{
    /// Slice the tensor along the specified dimension
    pub fn slice<const DIM: usize, const START: usize, const END: usize>(&self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::storage_uninit();
        B::slice4::<D0, D1, D2, D3, DIM, START, END>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
