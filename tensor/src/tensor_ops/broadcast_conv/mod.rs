mod naive_cpu;

use std::ops::{Add, Mul};
use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};

pub trait BroadcastConv3<T: Copy + Default>: Sized {
    /// Given an input of shape `[BATCH × (H×W)]` and a kernel of shape `[KH×KW]`,
    /// produce an output of shape `[BATCH × ((H - KH + 1) × (W - KW + 1))]`.
    fn conv3<const BATCH: usize, const H: usize, const W: usize, const KH: usize, const KW: usize>(
        input: &<Self as HasStorage<T, { BATCH * (H * W) }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { BATCH * ((H - KH + 1) * (W - KW + 1)) }>>::Storage,
    ) where
        Self: HasStorage<T, { BATCH * (H * W) }>
        + HasStorage<T, { KH * KW }>
        + HasStorage<T, { BATCH * ((H - KH + 1) * (W - KW + 1)) }>;
}

pub trait BroadcastConv4<T: Copy + Default>: Sized {
    /// Given an input of shape `[B0 × (B1 × (H×W))]` and a kernel `[KH×KW]`,
    /// produce an output of shape `[B0 × (B1 × ((H - KH + 1)×(W - KW + 1)))]`.
    fn conv4<
        const B0: usize,
        const B1: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
    >(
        input: &<Self as HasStorage<T, { B0 * (B1 * (H * W)) }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { B0 * (B1 * ((H - KH + 1) * (W - KW + 1))) }>>::Storage,
    ) where
        Self: HasStorage<T, { B0 * (B1 * (H * W)) }>
        + HasStorage<T, { KH * KW }>
        + HasStorage<T, { B0 * (B1 * ((H - KH + 1) * (W - KW + 1))) }>;
}

impl<T, const BATCH: usize, const H: usize, const W: usize, B> Tensor3<T, BATCH, H, W, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { BATCH * (H * W) }>,
{
    /// For a 3D tensor of shape [BATCH × H × W], convolve each H×W slice
    /// against the 2D kernel.  Returns a 3D tensor of shape
    /// [BATCH × (H-KH+1) × (W-KW+1)].
    pub fn convolve<const KH: usize, const KW: usize>(
        &self,
        kernel: &Tensor2<T, KH, KW, B>,
    ) -> Tensor3<T, BATCH, { H - KH + 1 }, { W - KW + 1 }, B>
    where
        B: BroadcastConv3<T> + HasStorage<T, { KH * KW }>
        + HasStorage<T, { BATCH * ((H - KH + 1) * (W - KW + 1)) }>,
    {
        let mut out = <B as HasStorage<T, { BATCH * ((H - KH + 1) * (W - KW + 1)) }>>::storage_uninit();
        B::conv3::<BATCH, H, W, KH, KW>(&self.storage, &kernel.storage, &mut out);
        Tensor3 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const B0: usize, const B1: usize, const H: usize, const W: usize, B> Tensor4<T, B0, B1, H, W, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { B0 * (B1 * (H * W)) }>,
{
    /// For a 4D tensor of shape [B0 × B1 × H × W], convolve each H×W slice
    /// (for every pair (i0,i1)) with the same KH×KW kernel.  The result is
    /// a new 4D tensor [B0 × B1 × (H-KH+1) × (W-KW+1)].
    pub fn convolve<const KH: usize, const KW: usize>(
        &self,
        kernel: &Tensor2<T, KH, KW, B>,
    ) -> Tensor4<T, B0, B1, { H - KH + 1 }, { W - KW + 1 }, B>
    where
        B: BroadcastConv4<T> + HasStorage<T, { KH * KW }>
        + HasStorage<T, { B0 * (B1 * ((H - KH + 1) * (W - KW + 1))) }>,
    {
        let mut out = <B as HasStorage<T, { B0 * (B1 * ((H - KH + 1) * (W - KW + 1))) }>>::storage_uninit();
        B::conv4::<B0, B1, H, W, KH, KW>(&self.storage, &kernel.storage, &mut out);
        Tensor4 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}