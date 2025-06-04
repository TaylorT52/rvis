//! NaiveCPU backend – reference “school‑book” implementation.
//! Works for any `T` that supports `Default + Add + Mul`.

use core::ops::{Add, Mul};

use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::matmul::MatMul;

impl<T> MatMul<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    fn matmul<const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * K }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { R * K }>,
    {
        // Get flat slices we can index into.
        let a = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { C * K }>>::as_slice(b);
        let out = <Self as HasStorage<T, { R * K }>>::as_mut_slice(out);

        // Triple‑nested loop (row‑major × col‑major – both are flat here).
        for r in 0..R {
            for k in 0..K {
                let mut acc = T::default();
                for c in 0..C {
                    acc = acc + a[r * C + c] * b[c * K + k];
                    // a[...] and b[...] are &T; Copy lets us use them as values.
                }
                out[r * K + k] = acc;
            }
        }
    }
}
