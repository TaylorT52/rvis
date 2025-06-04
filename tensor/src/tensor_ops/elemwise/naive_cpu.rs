use crate::storage::naive_cpu::NaiveCpu;
use crate::storage::HasStorage;
use crate::tensor_ops::elemwise::{ElemAdd, ElemMul, ElemSub};
use core::ops::{Add, Mul, Sub};

impl<T> ElemAdd<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T>,
{
    fn elem_add<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>,
    {
        let a = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { R * C }>>::as_slice(b);
        let dst = <Self as HasStorage<T, { R * C }>>::as_mut_slice(out);
        for i in 0..(R * C) {
            dst[i] = a[i] + b[i];
        }
    }
}

impl<T> ElemSub<T> for NaiveCpu
where
    T: Copy + Default + Sub<Output = T>,
{
    fn elem_sub<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>,
    {
        let a = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { R * C }>>::as_slice(b);
        let dst = <Self as HasStorage<T, { R * C }>>::as_mut_slice(out);
        for i in 0..(R * C) {
            dst[i] = a[i] - b[i];
        }
    }
}

impl<T> ElemMul<T> for NaiveCpu
where
    T: Copy + Default + Mul<Output = T>,
{
    fn elem_mul<const R: usize, const C: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        b: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }>,
    {
        let a = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let b = <Self as HasStorage<T, { R * C }>>::as_slice(b);
        let dst = <Self as HasStorage<T, { R * C }>>::as_mut_slice(out);
        for i in 0..(R * C) {
            dst[i] = a[i] * b[i];
        }
    }
}
