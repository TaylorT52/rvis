//! NaiveCPU backend – reference “school‑book” implementation.
//! Works for any `T` that supports `Default + Add + Mul`.

use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::matmul::MatMul;
use core::ops::{Add, Mul};

impl<T> MatMul<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    fn matmul<const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * K }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }> + HasStorage<T, { C * K }> + HasStorage<T, { R * K }>,
    {
        // Get flat slices we can index into.
        let a = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { C * K }>>::as_slice(b);
        let out = <Self as HasStorage<T, { R * K }>>::as_mut_slice(out);

        for r in 0..R {
            for k in 0..K {
                let mut acc = T::default();
                for c in 0..C {
                    acc = acc + a[r * C + c] * b[c * K + k];
                }
                out[r * K + k] = acc;
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use crate::storage::naive_cpu::NaiveCpu;
    use crate::tensor::Tensor2;
    use test::Bencher;

    #[bench]
    fn matmul_128x128(b: &mut Bencher) {
        const N: usize = 256;

        let a = Tensor2::<f32, N, N, NaiveCpu>::new([0.0; N * N]);
        let b_mat = Tensor2::<f32, N, N, NaiveCpu>::new([0.0; N * N]);

        b.iter(|| {
            // black_box stops LLVM from deleting the call
            test::black_box(a * b_mat);
        });
    }
}
