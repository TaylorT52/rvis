//! Naive “school‑book” reference implementation.

use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::conv::Conv2;
use core::ops::{Add, Mul};

impl<T> Conv2<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
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
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>,
    {
        let inp = <Self as HasStorage<T, { H * W }>>::as_slice(input);
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let out = <Self as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::as_mut_slice(output);

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        for i in 0..out_h {
            for j in 0..out_w {
                let mut acc = T::default();
                for ki in 0..KH {
                    for kj in 0..KW {
                        let hi = i * STRIDE + ki;
                        let wj = j * STRIDE + kj;
                        if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                            acc = acc + inp[(hi - PAD) * W + (wj - PAD)] * ker[ki * KW + kj];
                        }
                    }
                }
                out[i * out_w + j] = acc;
            }
        }
    }

    fn conv2_backward<
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        grad_output: &<Self as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::Storage,
        grad_input: &mut <Self as HasStorage<T, { H * W }>>::Storage,
    ) where
        Self: HasStorage<T, { H * W }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>,
    {
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let grad_out = <Self as HasStorage<T, { ((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1) }>>::as_slice(grad_output);
        let grad_in = <Self as HasStorage<T, { H * W }>>::as_mut_slice(grad_input);

        // zero initialize grad_in
        for v in grad_in.iter_mut() {
            *v = T::default();
        }

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        for i in 0..out_h {
            for j in 0..out_w {
                let go = grad_out[i * out_w + j];
                for ki in 0..KH {
                    for kj in 0..KW {
                        let hi = i * STRIDE + ki;
                        let wj = j * STRIDE + kj;
                        if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                            grad_in[(hi - PAD) * W + (wj - PAD)] =
                                grad_in[(hi - PAD) * W + (wj - PAD)] + ker[ki * KW + kj] * go;
                        }
                    }
                }
            }
        }
    }
}
