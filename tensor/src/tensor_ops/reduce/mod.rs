pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
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

pub trait Max3<T: Copy + Default + PartialOrd>: Sized {
    fn max_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;

    fn max_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>;

    fn max_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>;
}

pub trait Argmax3<T: Copy + Default + PartialOrd>: Sized {
    fn argmax_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D1 * D2 }>;

    fn argmax_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D0 * D2 }>;

    fn argmax_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D0 * D1 }>;
}

pub trait Mean4<T: Float + Default>: Sized {
    fn mean_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        T: Float,
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>;
}

pub trait Max4<T: Copy + Default + PartialOrd>: Sized {
    fn max_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>;
}

pub trait Argmax4<T: Copy + Default + PartialOrd>: Sized {
    fn argmax_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<usize, { D0 * D1 }>;
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

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, B>
    Tensor4<T, D0, D1, D2, D3, B>
where
    T: Float + Default,
    B: Mean4<T> + HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * (D2 * D3))]:,
{
    #[inline]
    pub fn mean_axis23(self) -> Tensor2<T, D0, D1, B> {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::mean_axis23::<D0, D1, D2, D3>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, B>
    Tensor4<T, D0, D1, D2, D3, B>
where
    T: Copy + Default + PartialOrd,
    B: Max4<T> + HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * (D2 * D3))]:,
{
    #[inline]
    pub fn max_axis23(self) -> Tensor2<T, D0, D1, B> {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::max_axis23::<D0, D1, D2, D3>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, const D3: usize, B>
    Tensor4<T, D0, D1, D2, D3, B>
where
    T: Copy + Default + PartialOrd,
    B: Argmax4<T>
        + HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
        + HasStorage<usize, { D0 * D1 }>,
    [(); D0 * (D1 * (D2 * D3))]:,
{
    #[inline]
    pub fn argmax_axis23(self) -> Tensor2<usize, D0, D1, B> {
        let mut out = <B as HasStorage<usize, { D0 * D1 }>>::storage_uninit();
        B::argmax_axis23::<D0, D1, D2, D3>(&self.storage, &mut out);
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

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + PartialOrd,
    B: Max3<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>
        + HasStorage<T, { D0 * D2 }>
        + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn max_axis0(self) -> Tensor2<T, D1, D2, B> {
        let mut out = <B as HasStorage<T, { D1 * D2 }>>::storage_uninit();
        B::max_axis0::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn max_axis1(self) -> Tensor2<T, D0, D2, B> {
        let mut out = <B as HasStorage<T, { D0 * D2 }>>::storage_uninit();
        B::max_axis1::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn max_axis2(self) -> Tensor2<T, D0, D1, B> {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::max_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + PartialOrd,
    B: Argmax3<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<usize, { D1 * D2 }>
        + HasStorage<usize, { D0 * D2 }>
        + HasStorage<usize, { D0 * D1 }>,
    [(); D0 * (D1 * D2)]:,
{
    #[inline]
    pub fn argmax_axis0(self) -> Tensor2<usize, D1, D2, B> {
        let mut out = <B as HasStorage<usize, { D1 * D2 }>>::storage_uninit();
        B::argmax_axis0::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn argmax_axis1(self) -> Tensor2<usize, D0, D2, B> {
        let mut out = <B as HasStorage<usize, { D0 * D2 }>>::storage_uninit();
        B::argmax_axis1::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn argmax_axis2(self) -> Tensor2<usize, D0, D1, B> {
        let mut out = <B as HasStorage<usize, { D0 * D1 }>>::storage_uninit();
        B::argmax_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

}

impl<T, const D0: usize, const D1: usize, const D2: usize, B> Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default,
    B: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>,
    [(); D0 * (D1 * D2)]:,
{
    /// Reduce over the last dimension, treating the first two as batch dims.
    #[inline]
    pub fn mean_batches(self) -> Tensor2<T, D0, D1, B>
    where
        T: Float,
        B: Mean3<T>,
    {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::mean_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn max_batches(self) -> Tensor2<T, D0, D1, B>
    where
        T: PartialOrd,
        B: Max3<T>,
    {
        let mut out = <B as HasStorage<T, { D0 * D1 }>>::storage_uninit();
        B::max_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn argmax_batches(self) -> Tensor2<usize, D0, D1, B>
    where
        T: PartialOrd,
        B: Argmax3<T> + HasStorage<usize, { D0 * D1 }>,
    {
        let mut out = <B as HasStorage<usize, { D0 * D1 }>>::storage_uninit();
        B::argmax_axis2::<D0, D1, D2>(&self.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
