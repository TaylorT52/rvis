use crate::storage::HasStorage;
use crate::tensor::Tensor2;
use core::ops::{Add, Mul};

pub mod naive_cpu;

pub trait Conv2<T: Copy + Default>: Sized {
    fn conv2<
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        input: &<Self as HasStorage<T, { H * W }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output:
            &mut <Self as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::Storage,
    ) where
        Self: HasStorage<T, { H * W }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>;

    fn conv2_backward<
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        grad_output:
            &<Self as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::Storage,
        grad_input: &mut <Self as HasStorage<T, { H * W }>>::Storage,
    ) where
        Self: HasStorage<T, { H * W }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>;
}

impl<T, const H: usize, const W: usize, B> Tensor2<T, H, W, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { H * W }>,
{
    pub fn convolve<const KH: usize, const KW: usize, const STRIDE: usize, const PAD: usize>(
        &self,
        kernel: &Tensor2<T, KH, KW, B>,
    ) -> Tensor2<T, { (H + 2 * PAD - KH) / STRIDE + 1 }, { (W + 2 * PAD - KW) / STRIDE + 1 }, B>
    where
        B: Conv2<T>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>,
    {
        let mut out = <B as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::storage_uninit();
        B::conv2::<H, W, KH, KW, STRIDE, PAD>(&self.storage, &kernel.storage, &mut out);
        Tensor2 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }

    pub fn conv2_backward<const KH: usize, const KW: usize, const STRIDE: usize, const PAD: usize>(
        grad_output: &Tensor2<T, { (H + 2 * PAD - KH) / STRIDE + 1 }, { (W + 2 * PAD - KW) / STRIDE + 1 }, B>,
        kernel: &Tensor2<T, KH, KW, B>,
    ) -> Self
    where
        B: Conv2<T>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>,
    {
        let mut grad_input = <B as HasStorage<T, { H * W }>>::storage_uninit();
        B::conv2_backward::<H, W, KH, KW, STRIDE, PAD>(
            &kernel.storage,
            &grad_output.storage,
            &mut grad_input,
        );
        Tensor2 {
            storage: grad_input,
            _p: core::marker::PhantomData,
        }
    }
}
