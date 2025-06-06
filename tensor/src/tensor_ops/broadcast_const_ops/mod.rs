pub mod naive_cpu;

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
use core::ops::{Add, Div, Mul, Sub};

pub trait BroadcastConstAdd<T: Copy + Default>: Sized {
    fn add43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>;

    fn add42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>;

    fn add32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;
}

pub trait BroadcastConstSub<T: Copy + Default>: Sized {
    fn sub43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>;

    fn sub42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>;

    fn sub32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;
}

pub trait BroadcastConstMul<T: Copy + Default>: Sized {
    fn mul43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>;

    fn mul42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>;

    fn mul32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;
}

pub trait BroadcastConstDiv<T: Copy + Default>: Sized {
    fn div43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>;

    fn div42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>;

    fn div32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>;
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Add<Tensor3<T, D1, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Add<Output = T>,
    B: BroadcastConstAdd<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D1 * (D3 * D4) }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn add(self, rhs: Tensor3<T, D1, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::add43::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Sub<Tensor3<T, D1, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Sub<Output = T>,
    B: BroadcastConstSub<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D1 * (D3 * D4) }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn sub(self, rhs: Tensor3<T, D1, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::sub43::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Mul<Tensor3<T, D1, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: BroadcastConstMul<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D1 * (D3 * D4) }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn mul(self, rhs: Tensor3<T, D1, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::mul43::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Div<Tensor3<T, D1, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Div<Output = T>,
    B: BroadcastConstDiv<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D1 * (D3 * D4) }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn div(self, rhs: Tensor3<T, D1, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::div43::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Add<Tensor2<T, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Add<Output = T>,
    B: BroadcastConstAdd<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D3 * D4 }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn add(self, rhs: Tensor2<T, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::add42::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Sub<Tensor2<T, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Sub<Output = T>,
    B: BroadcastConstSub<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D3 * D4 }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn sub(self, rhs: Tensor2<T, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::sub42::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<'a, T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Mul<&'a Tensor2<T, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: BroadcastConstMul<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D3 * D4 }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn mul(self, rhs: &Tensor2<T, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::mul42::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D3: usize, const D4: usize, B>
    Div<Tensor2<T, D3, D4, B>> for Tensor4<T, D0, D1, D3, D4, B>
where
    T: Copy + Default + Div<Output = T>,
    B: BroadcastConstDiv<T>
        + HasStorage<T, { D0 * (D1 * (D3 * D4)) }>
        + HasStorage<T, { D3 * D4 }>,
{
    type Output = Tensor4<T, D0, D1, D3, D4, B>;
    #[inline]
    fn div(self, rhs: Tensor2<T, D3, D4, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::storage_uninit();
        B::div42::<D0, D1, D3, D4>(&self.storage, &rhs.storage, &mut out);
        Tensor4 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B>
    Add<Tensor2<T, D1, D2, B>> for Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Add<Output = T>,
    B: BroadcastConstAdd<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>,
{
    type Output = Tensor3<T, D0, D1, D2, B>;
    #[inline]
    fn add(self, rhs: Tensor2<T, D1, D2, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::add32::<D0, D1, D2>(&self.storage, &rhs.storage, &mut out);
        Tensor3 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B>
    Sub<Tensor2<T, D1, D2, B>> for Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Sub<Output = T>,
    B: BroadcastConstSub<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>,
{
    type Output = Tensor3<T, D0, D1, D2, B>;
    #[inline]
    fn sub(self, rhs: Tensor2<T, D1, D2, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::sub32::<D0, D1, D2>(&self.storage, &rhs.storage, &mut out);
        Tensor3 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<'a, T, const D0: usize, const D1: usize, const D2: usize, B>
    Mul<&'a Tensor2<T, D1, D2, B>> for Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: BroadcastConstMul<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>,
{
    type Output = Tensor3<T, D0, D1, D2, B>;
    #[inline]
    fn mul(self, rhs: &Tensor2<T, D1, D2, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::mul32::<D0, D1, D2>(&self.storage, &rhs.storage, &mut out);
        Tensor3 { storage: out, _p: core::marker::PhantomData }
    }
}

impl<T, const D0: usize, const D1: usize, const D2: usize, B>
    Div<Tensor2<T, D1, D2, B>> for Tensor3<T, D0, D1, D2, B>
where
    T: Copy + Default + Div<Output = T>,
    B: BroadcastConstDiv<T>
        + HasStorage<T, { D0 * (D1 * D2) }>
        + HasStorage<T, { D1 * D2 }>,
{
    type Output = Tensor3<T, D0, D1, D2, B>;
    #[inline]
    fn div(self, rhs: Tensor2<T, D1, D2, B>) -> Self::Output {
        let mut out = <B as HasStorage<T, { D0 * (D1 * D2) }>>::storage_uninit();
        B::div32::<D0, D1, D2>(&self.storage, &rhs.storage, &mut out);
        Tensor3 { storage: out, _p: core::marker::PhantomData }
    }
}

