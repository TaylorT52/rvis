pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3};
use core::ops::Add;
use num_traits::Float;

pub trait Sum3<T: Copy + Default>: Sized {
    fn sum_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;

    fn sum_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>;

    fn sum_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        T: Add<Output = T>,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>;
}

pub trait Mean3<T: Float + Default>: Sized {
    fn mean_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        T: Float,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;

    fn mean_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        T: Float,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>;

    fn mean_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        T: Float,
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>;
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Add<Output = T>,
    B: Sum3<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>
        + HasStorage<T, { D0 * D2 }>
        + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn sum_axis0(self) -> Tensor2<T, D1, D2, B> {
        let mut out = <B as HasStorage<T, { D1 * D2 }>>::storage_uninit();
        B::sum_axis0::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn sum_axis1(self) -> Tensor2<T, D0, D2, B> {
        let mut out = <B as HasStorage<T, { D0 * D2 }>>::storage_uninit();
        B::sum_axis1::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn sum_axis2(self) -> Tensor2<T, D0, D1, B> {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::sum_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Float + Default,
    B: Mean3<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>
        + HasStorage<T, { D0 * D2 }>
        + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn mean_axis0(self) -> Tensor2<T, D1, D2, B> {
        let mut out = <B as HasStorage<T, { D1 * D2 }>>::storage_uninit();
        B::mean_axis0::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn mean_axis1(self) -> Tensor2<T, D0, D2, B> {
        let mut out = <B as HasStorage<T, { D0 * D2 }>>::storage_uninit();
        B::mean_axis1::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn mean_axis2(self) -> Tensor2<T, D0, D1, B> {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::mean_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
