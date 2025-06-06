use crate::storage::naive_cpu::NaiveCpu;
use crate::storage::HasStorage;
use crate::tensor_ops::broadcast_const_ops::{
    BroadcastConstAdd, BroadcastConstDiv, BroadcastConstMul, BroadcastConstSub,
};

impl<T> BroadcastConstAdd<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Add<Output = T>,
{
    fn add43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * (D3 * D4) }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * (D3 * D4));
            for i in 0..(D1 * (D3 * D4)) {
                o[base + i] = a[base + i] + b[i];
            }
        }
    }

    fn add42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D3 * D4 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let base = (i0 * D1 + i1) * (D3 * D4);
                for i in 0..(D3 * D4) {
                    o[base + i] = a[base + i] + b[i];
                }
            }
        }
    }

    fn add32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * D2 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * D2);
            for i in 0..(D1 * D2) {
                o[base + i] = a[base + i] + b[i];
            }
        }
    }
}

impl<T> BroadcastConstSub<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Sub<Output = T>,
{
    fn sub43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * (D3 * D4) }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * (D3 * D4));
            for i in 0..(D1 * (D3 * D4)) {
                o[base + i] = a[base + i] - b[i];
            }
        }
    }

    fn sub42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D3 * D4 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let base = (i0 * D1 + i1) * (D3 * D4);
                for i in 0..(D3 * D4) {
                    o[base + i] = a[base + i] - b[i];
                }
            }
        }
    }

    fn sub32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * D2 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * D2);
            for i in 0..(D1 * D2) {
                o[base + i] = a[base + i] - b[i];
            }
        }
    }
}

impl<T> BroadcastConstMul<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Mul<Output = T>,
{
    fn mul43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * (D3 * D4) }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * (D3 * D4));
            for i in 0..(D1 * (D3 * D4)) {
                o[base + i] = a[base + i] * b[i];
            }
        }
    }

    fn mul42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D3 * D4 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let base = (i0 * D1 + i1) * (D3 * D4);
                for i in 0..(D3 * D4) {
                    o[base + i] = a[base + i] * b[i];
                }
            }
        }
    }

    fn mul32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * D2 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * D2);
            for i in 0..(D1 * D2) {
                o[base + i] = a[base + i] * b[i];
            }
        }
    }
}

impl<T> BroadcastConstDiv<T> for NaiveCpu
where
    T: Copy + Default + core::ops::Div<Output = T>,
{
    fn div43<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * (D3 * D4) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D1 * (D3 * D4) }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * (D3 * D4) }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * (D3 * D4));
            for i in 0..(D1 * (D3 * D4)) {
                o[base + i] = a[base + i] / b[i];
            }
        }
    }

    fn div42<const D0: usize, const D1: usize, const D3: usize, const D4: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
        b: &<Self as HasStorage<T, { D3 * D4 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D3 * D4)) }> + HasStorage<T, { D3 * D4 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D3 * D4 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * (D3 * D4)) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let base = (i0 * D1 + i1) * (D3 * D4);
                for i in 0..(D3 * D4) {
                    o[base + i] = a[base + i] / b[i];
                }
            }
        }
    }

    fn div32<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        b: &<Self as HasStorage<T, { D1 * D2 }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let a = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let b = <Self as HasStorage<T, { D1 * D2 }>>::as_slice(b);
        let o = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            let base = i0 * (D1 * D2);
            for i in 0..(D1 * D2) {
                o[base + i] = a[base + i] / b[i];
            }
        }
    }
}

