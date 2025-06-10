//! Tensor reshape operations.
//!
//! Backend implementers should implement [`Reshape`] for their backend.

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};

/// Trait for backends that support reshaping tensors while keeping the same data order.
pub trait Reshape<T: Copy + Default>: Sized {
    /// Reshape a 2D tensor into another 2D tensor of the same total size.
    fn reshape22<const R: usize, const C: usize, const NR: usize, const NC: usize>(
        src: &<Self as HasStorage<T, { R * C }>>::Storage,
        dst: &mut <Self as HasStorage<T, { NR * NC }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }> + HasStorage<T, { NR * NC }>;

    /// Reshape a 3D tensor into a 2D tensor of the same total size.
    fn reshape32<const D0: usize, const D1: usize, const D2: usize, const R: usize, const C: usize>(
        src: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { R * C }>;

    /// Reshape a 3D tensor into another 3D tensor of the same total size.
    fn reshape33<const D0: usize, const D1: usize, const D2: usize, const ND0: usize, const ND1: usize, const ND2: usize>(
        src: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { ND0 * (ND1 * ND2) }>;

    /// Reshape a 4D tensor into another 4D tensor of the same total size.
    fn reshape44<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const ND0: usize,
        const ND1: usize,
        const ND2: usize,
        const ND3: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>;

    /// Reshape a 4D tensor into a 3D tensor of the same total size.
    fn reshape43<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const ND0: usize,
        const ND1: usize,
        const ND2: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { ND0 * (ND1 * ND2) }>;

    /// Reshape a 4D tensor into a 2D tensor of the same total size.
    fn reshape42<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const R: usize,
        const C: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { R * C }>;
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: Copy + Default,
    B: Reshape<T> + HasStorage<T, { R * C }>,
{
    /// Reshape this 2D tensor to a new 2D tensor with shape `[NR, NC]`.
    pub fn reshape2<const NR: usize, const NC: usize>(self) -> Tensor2<T, NR, NC, B>
    where
        B: HasStorage<T, { NR * NC }>,
    {
        let mut out = <B as HasStorage<T, { NR * NC }>>::storage_uninit();
        B::reshape22::<R, C, NR, NC>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default,
    B: Reshape<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    /// Reshape this 3D tensor to a 2D tensor `[R, C]`.
    pub fn reshape2<const R: usize, const C: usize>(self) -> Tensor2<T, R, C, B>
    where
        B: HasStorage<T, { R * C }>,
    {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::reshape32::<D0, D1, D2, R, C>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    /// Reshape this 3D tensor to another 3D tensor `[ND0, ND1, ND2]`.
    pub fn reshape3<const ND0: usize, const ND1: usize, const ND2: usize>(
        self,
    ) -> Tensor3<T, ND0, ND1, ND2, B>
    where
        B: HasStorage<T, { ND0 * (ND1 * ND2) }>,
    {
        let mut out = <B as HasStorage<T, { ND0 * (ND1 * ND2) }>>::storage_uninit();
        B::reshape33::<D0, D1, D2, ND0, ND1, ND2>(&self.storage, &mut out);
        Tensor3 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<
        T,
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        B,
    > Tensor4<T, D0, D1, D2, D3, B>
where
    T: Copy + Default,
    B: Reshape<T> + HasStorage<T, { D0 * (D1 * (D2 * D3)) }>,
    [(); D0 * (D1 * (D2 * D3))]:,
{
    /// Reshape this 4D tensor to another 4D tensor `[ND0, ND1, ND2, ND3]`.
    pub fn reshape4<
        const ND0: usize,
        const ND1: usize,
        const ND2: usize,
        const ND3: usize,
    >(
        self,
    ) -> Tensor4<T, ND0, ND1, ND2, ND3, B>
    where
        B: HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>,
    {
        let mut out = <B as HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>>::storage_uninit();
        B::reshape44::<D0, D1, D2, D3, ND0, ND1, ND2, ND3>(&self.storage, &mut out);
        Tensor4 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    /// Reshape this 4D tensor to a 3D tensor `[ND0, ND1, ND2]`.
    pub fn reshape3<const ND0: usize, const ND1: usize, const ND2: usize>(
        self,
    ) -> Tensor3<T, ND0, ND1, ND2, B>
    where
        B: HasStorage<T, { ND0 * (ND1 * ND2) }>,
    {
        let mut out = <B as HasStorage<T, { ND0 * (ND1 * ND2) }>>::storage_uninit();
        B::reshape43::<D0, D1, D2, D3, ND0, ND1, ND2>(&self.storage, &mut out);
        Tensor3 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    /// Reshape this 4D tensor to a 2D tensor `[R, C]`.
    pub fn reshape2<const R: usize, const C: usize>(self) -> Tensor2<T, R, C, B>
    where
        B: HasStorage<T, { R * C }>,
    {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::reshape42::<D0, D1, D2, D3, R, C>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
