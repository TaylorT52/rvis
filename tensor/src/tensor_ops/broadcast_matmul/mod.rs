//! Broadcasted matrix-multiplication operators and backend traits.
//!
//! This module adds compile-time shape‑checked matmul for 3‑D and 4‑D tensors
//! that broadcast over one or two batch dimensions.

#![allow(clippy::needless_range_loop)]

use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor3, Tensor4};
use std::ops::{Add, Mul};

pub trait BroadcastMatMul3<T: Copy + Default>: Sized {
    fn matmul3<const BATCH: usize, const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<T, { BATCH * (R * C) }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { BATCH * (R * K) }>>::Storage,
    ) where
        Self: HasStorage<T, { BATCH * (R * C) }>
            + HasStorage<T, { C * K }>
            + HasStorage<T, { BATCH * (R * K) }>;
}

pub trait BroadcastMatMul4<T: Copy + Default>: Sized {
    fn matmul4<const B0: usize, const B1: usize, const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<T, { B0 * (B1 * (R * C)) }>>::Storage,
        b: &<Self as HasStorage<T, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<T, { B0 * (B1 * (R * K)) }>>::Storage,
    ) where
        Self: HasStorage<T, { B0 * (B1 * (R * C)) }>
            + HasStorage<T, { C * K }>
            + HasStorage<T, { B0 * (B1 * (R * K)) }>;
}

impl<T, const BATCH: usize, const R: usize, const C: usize, const K: usize, B>
    Mul<Tensor2<T, C, K, B>> for Tensor3<T, BATCH, R, C, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { BATCH * (R * C) }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { BATCH * (R * K) }>
        + BroadcastMatMul3<T>,
{
    type Output = Tensor3<T, BATCH, R, K, B>;

    #[inline]
    fn mul(self, rhs: Tensor2<T, C, K, B>) -> Self::Output {
        let mut out: <B as HasStorage<T, { BATCH * (R * K) }>>::Storage =
            <B as HasStorage<T, { BATCH * (R * K) }>>::storage_uninit();

        B::matmul3::<BATCH, R, C, K>(&self.storage, &rhs.storage, &mut out);

        Tensor3 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}

impl<T, const B0: usize, const B1: usize, const R: usize, const C: usize, const K: usize, B>
    Mul<Tensor2<T, C, K, B>> for Tensor4<T, B0, B1, R, C, B>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    B: HasStorage<T, { B0 * (B1 * (R * C)) }>
        + HasStorage<T, { C * K }>
        + HasStorage<T, { B0 * (B1 * (R * K)) }>
        + BroadcastMatMul4<T>,
{
    type Output = Tensor4<T, B0, B1, R, K, B>;

    #[inline]
    fn mul(self, rhs: Tensor2<T, C, K, B>) -> Self::Output {
        let mut out: <B as HasStorage<T, { B0 * (B1 * (R * K)) }>>::Storage =
            <B as HasStorage<T, { B0 * (B1 * (R * K)) }>>::storage_uninit();

        B::matmul4::<B0, B1, R, C, K>(&self.storage, &rhs.storage, &mut out);

        Tensor4 {
            storage: out,
            _p: core::marker::PhantomData,
        }
    }
}
