//! Element-wise ReLU (rectified linear unit) for tensors.
//!
//! Backend implementers should implement [`Relu`].

pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
use core::cmp::PartialOrd;

/// Backend trait for element-wise ReLU.
pub trait Relu<T: Copy + Default + PartialOrd>: Sized {
    fn relu<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>;

    fn relu_backward<const N: usize>(
        input: &<Self as HasStorage<T, N>>::Storage,
        grad_output: &<Self as HasStorage<T, N>>::Storage,
        grad_input: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>;
}

impl<T, const R: usize, const C: usize, B> Tensor2<T, R, C, B>
where
    T: Copy + Default + PartialOrd,
    B: Relu<T> + HasStorage<T, { R * C }>,
{
    #[inline]
    pub fn relu(self) -> Self {
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::relu::<{ R * C }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + PartialOrd,
    B: Relu<T> + HasStorage<T, { D0 * (D1 * D2) }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn relu(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::relu::<{ D0 * (D1 * D2) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + PartialOrd,
    B: Relu<T> + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>,
    [(); D0 * (D1 * (D3 * D4))]:,
{
    #[inline]
    pub fn relu(self) -> Self {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::relu::<{ D0 * (D1 * (D3 * D4)) }>(&self.storage, &mut out);
        Self {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn relu_backward(
        input: &Self,
        grad_output: &Self,
    ) -> Self {
        let mut grad_input = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::relu_backward::<{ D0 * (D1 * (D3 * D4)) }>(
            &input.storage,
            &grad_output.storage,
            &mut grad_input,
        );
        Self {
            storage: grad_input,
            _p: core::marker::PhantomData,
        }
    }
}
