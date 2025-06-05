//! Naive “school‑book” reference implementation.

use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::conv::Conv2;
use core::ops::{Add, Mul};

impl<T> Conv2<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    fn conv2<const H: usize, const W: usize, const KH: usize, const KW: usize>(
        input: &<Self as HasStorage<T, { H * W }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>>::Storage,
    ) where
        Self: HasStorage<T, { H * W }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>,
    {
        let inp = <Self as HasStorage<T, { H * W }>>::as_slice(input);
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let out = <Self as HasStorage<T, { (H - KH + 1) * (W - KW + 1) }>>::as_mut_slice(output);

        let out_h = H - KH + 1;
        let out_w = W - KW + 1;

        for i in 0..out_h {
            for j in 0..out_w {
                let mut acc = T::default();
                for ki in 0..KH {
                    for kj in 0..KW {
                        acc = acc + inp[(i + ki) * W + (j + kj)] * ker[ki * KW + kj];
                    }
                }
                out[i * out_w + j] = acc;
            }
        }
    }
}
