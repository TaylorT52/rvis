//! Element-wise exponential (e^x) for tensors.
//!
//! Backend implementers should implement [`Exp`].

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};

/// Trait for types that support the exponential function.
pub trait ExpElem: Copy + Default {
    /// Returns `e^self`.
    fn exp(self) -> Self;
}

impl ExpElem for f32 {
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
}

impl ExpElem for f64 {
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
}

/// Backend trait for element-wise exponential.
pub trait Exp<T: ExpElem>: Sized {
    fn exp<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>;
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: ExpElem,
    B: Exp<T> + HasStorage<T, { R * C }>,
{
    #[inline]
    pub fn exp(self) -> Self {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::exp::<{ R * C }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: ExpElem,
    B: Exp<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn exp(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::exp::<{ D0 * (D1 * D2) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Tensor4<T, D0, D1, D3, D4, B>
where
    T: ExpElem,
    B: Exp<T> + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>,
    [(); D0 * (D1 * (D3 * D4))]:,
{
    #[inline]
    pub fn exp(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::exp::<{ D0 * (D1 * (D3 * D4)) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
