use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::broadcast_matmul::{BroadcastMatMul3, BroadcastMatMul4};

impl<T> BroadcastMatMul3<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    fn matmul3<
        const BATCH: usize,
        const R: usize,
        const C: usize,
        const K: usize,
    >(
        a: &<Self as HasStorage<T, { BATCH * R * C }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { BATCH * R * K }>>::Storage,
    ) where
        Self: HasStorage<T, { BATCH * R * C }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { BATCH * R * K }>,
    {
        let a = <Self as HasStorage<T, { BATCH * R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { C * K }>>::as_slice(b);
        let o = <Self as HasStorage<T, { BATCH * R * K }>>::as_mut_slice(out);

        for batch in 0..BATCH {
            for r in 0..R {
                for k in 0..K {
                    let mut acc = T::default();
                    for c in 0..C {
                        acc = acc + a[batch * R * C + r * C + c] * b[c * K + k];
                    }
                    o[batch * R * K + r * K + k] = acc;
                }
            }
        }
    }
}

impl<T> BroadcastMatMul4<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    fn matmul4<
        const B0: usize,
        const B1: usize,
        const R: usize,
        const C: usize,
        const K: usize,
    >(
        a: &<Self as HasStorage<T, { B0 * B1 * R * C }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { B0 * B1 * R * K }>>::Storage,
    ) where
        Self: HasStorage<T, { B0 * B1 * R * C }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { B0 * B1 * R * K }>,
    {
        let a = <Self as HasStorage<T, { B0 * B1 * R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { C * K }>>::as_slice(b);
        let o = <Self as HasStorage<T, { B0 * B1 * R * K }>>::as_mut_slice(out);

        for i0 in 0..B0 {
            for i1 in 0..B1 {
                for r in 0..R {
                    for k in 0..K {
                        let mut acc = T::default();
                        for c in 0..C {
                            let idx_a = (((i0 * B1 + i1) * R + r) * C) + c;
                            acc = acc + a[idx_a] * b[c * K + k];
                        }
                        let idx_o = (((i0 * B1 + i1) * R + r) * K) + k;
                        o[idx_o] = acc;
                    }
                }
            }
        }
    }
}