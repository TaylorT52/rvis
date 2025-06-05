pub mod naive_cpu;

use std::ops::{Add, Mul};
use crate::storage::HasStorage;
use crate::tensor::{Tensor2, Tensor4};

pub trait BroadcastMatMul3<T: Copy + Default>: Sized {
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
        + HasStorage<T, { BATCH * R * K }>;
}

pub trait BroadcastMatMul4<T: Copy + Default>: Sized {
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
        + HasStorage<T, { B0 * B1 * R * K }>;
}
