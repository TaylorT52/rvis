use crate::storage::HasStorage;
use crate::tensor::Tensor2;
use core::ops::{Add, Mul};

pub mod naive_cpu;

pub trait Conv2<T: Copy + Default>: Sized {
    fn conv2<const H: usize, const W: usize, const KH: usize, const KW: usize>(
        input: &<Self as HasStorage<T, { H * W }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>>::Storage,
    ) where
        Self: HasStorage<T, { H * W }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>;
}

impl<T, const H: usize, const W: usize, B> Tensor2<T, H, W, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { H * W }>,
{
    pub fn convolve<const KH: usize, const KW: usize>(
        &self,
        kernel: &Tensor2<T, KH, KW, B>,
    ) -> Tensor2<T, { H - KH + 1 }, { W - KW + 1 }, B>
    where
        B: Conv2<T> + HasStorage<T, { KH * KW }> + HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>,
    {
        let mut out = <B as HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>>::storage_uninit();
        B::conv2::<H, W, KH, KW>(&self.storage, &kernel.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
