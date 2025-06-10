use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::broadcast_conv::{BroadcastConv3, BroadcastConv4};
use std::ops::{Add, Mul};

impl<T> BroadcastConv3<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    fn conv3<
        const BATCH: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        input: &<Self as HasStorage<T, { BATCH * (H * W) }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>>::Storage,
    ) where
        Self: HasStorage<T, { BATCH * (H * W) }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>,
    {
        // Convert storages into flat slices:
        let inp = <Self as HasStorage<T, { BATCH * (H * W) }>>::as_slice(input);
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let out = <Self as HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>>::as_mut_slice(output);

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        // For each batch, run a 2D conv on that H×W slice.
        for batch in 0..BATCH {
            let base_in = batch * (H * W);
            let base_out = batch * (out_h * out_w);

            for i in 0..out_h {
                for j in 0..out_w {
                    let mut acc = T::default();

                    // sum over the KH×KW window
                    for ki in 0..KH {
                        for kj in 0..KW {
                            let hi = i * STRIDE + ki;
                            let wj = j * STRIDE + kj;
                            if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                                let idx_in = base_in + (hi - PAD) * W + (wj - PAD);
                                let idx_ker = ki * KW + kj;
                                acc = acc + inp[idx_in] * ker[idx_ker];
                            }
                        }
                    }

                    let idx_out = base_out + i * out_w + j;
                    out[idx_out] = acc;
                }
            }
        }
    }

    fn conv3_backward<
        const BATCH: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        grad_output: &<Self as HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>>::Storage,
        grad_input: &mut <Self as HasStorage<T, { BATCH * (H * W) }>>::Storage,
    ) where
        Self: HasStorage<T, { BATCH * (H * W) }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>,
    {
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let grad_out = <Self as HasStorage<T, { BATCH * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1)) }>>::as_slice(grad_output);
        let grad_in = <Self as HasStorage<T, { BATCH * (H * W) }>>::as_mut_slice(grad_input);

        for v in grad_in.iter_mut() {
            *v = T::default();
        }

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        for batch in 0..BATCH {
            let base_in = batch * (H * W);
            let base_out = batch * (out_h * out_w);
            for i in 0..out_h {
                for j in 0..out_w {
                    let go = grad_out[base_out + i * out_w + j];
                    for ki in 0..KH {
                        for kj in 0..KW {
                            let hi = i * STRIDE + ki;
                            let wj = j * STRIDE + kj;
                            if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                                grad_in[base_in + (hi - PAD) * W + (wj - PAD)] =
                                    grad_in[base_in + (hi - PAD) * W + (wj - PAD)] + ker[ki * KW + kj] * go;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<T> BroadcastConv4<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    fn conv4<
        const B0: usize,
        const B1: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        input: &<Self as HasStorage<T, { B0 * (B1 * (H * W)) }>>::Storage,
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        output: &mut <Self as HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>>::Storage,
    ) where
        Self: HasStorage<T, { B0 * (B1 * (H * W)) }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>,
    {
        let inp = <Self as HasStorage<T, { B0 * (B1 * (H * W)) }>>::as_slice(input);
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let out =
            <Self as HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>>::as_mut_slice(output);

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        // Loop over both batch dims:
        for i0 in 0..B0 {
            for i1 in 0..B1 {
                // Compute the slice base for input and output:
                let base_in = ((i0 * B1 + i1) * (H * W)) as usize;
                let base_out = ((i0 * B1 + i1) * (out_h * out_w)) as usize;

                for i in 0..out_h {
                    for j in 0..out_w {
                        let mut acc = T::default();

                        // convolve over the KH×KW window within this (H×W) slice
                        for ki in 0..KH {
                            for kj in 0..KW {
                                let hi = i * STRIDE + ki;
                                let wj = j * STRIDE + kj;
                                if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                                    let idx_in = base_in + (hi - PAD) * W + (wj - PAD);
                                    let idx_ker = ki * KW + kj;
                                    acc = acc + inp[idx_in] * ker[idx_ker];
                                }
                            }
                        }

                        let idx_out = base_out + i * out_w + j;
                        out[idx_out] = acc;
                    }
                }
            }
        }
    }

    fn conv4_backward<
        const B0: usize,
        const B1: usize,
        const H: usize,
        const W: usize,
        const KH: usize,
        const KW: usize,
        const STRIDE: usize,
        const PAD: usize,
    >(
        kernel: &<Self as HasStorage<T, { KH * KW }>>::Storage,
        grad_output: &<Self as HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>>::Storage,
        grad_input: &mut <Self as HasStorage<T, { B0 * (B1 * (H * W)) }>>::Storage,
    ) where
        Self: HasStorage<T, { B0 * (B1 * (H * W)) }>
            + HasStorage<T, { KH * KW }>
            + HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>,
    {
        let ker = <Self as HasStorage<T, { KH * KW }>>::as_slice(kernel);
        let grad_out = <Self as HasStorage<T, { B0 * (B1 * (((H + 2 * PAD - KH) / STRIDE + 1) * ((W + 2 * PAD - KW) / STRIDE + 1))) }>>::as_slice(grad_output);
        let grad_in = <Self as HasStorage<T, { B0 * (B1 * (H * W)) }>>::as_mut_slice(grad_input);

        for v in grad_in.iter_mut() {
            *v = T::default();
        }

        let out_h = (H + 2 * PAD - KH) / STRIDE + 1;
        let out_w = (W + 2 * PAD - KW) / STRIDE + 1;

        for i0 in 0..B0 {
            for i1 in 0..B1 {
                let base_in = (i0 * B1 + i1) * (H * W);
                let base_out = (i0 * B1 + i1) * (out_h * out_w);
                for i in 0..out_h {
                    for j in 0..out_w {
                        let go = grad_out[base_out + i * out_w + j];
                        for ki in 0..KH {
                            for kj in 0..KW {
                                let hi = i * STRIDE + ki;
                                let wj = j * STRIDE + kj;
                                if hi >= PAD && hi < H + PAD && wj >= PAD && wj < W + PAD {
                                    grad_in[base_in + (hi - PAD) * W + (wj - PAD)] =
                                        grad_in[base_in + (hi - PAD) * W + (wj - PAD)] + ker[ki * KW + kj] * go;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
