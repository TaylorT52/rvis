use crate::storage::naive_cpu::NaiveCpu;
use crate::storage::HasStorage;
use crate::tensor_ops::reshape::Reshape;

impl<T> Reshape<T> for NaiveCpu
where
    T: Copy + Default,
{
    fn reshape22<const R: usize, const C: usize, const NR: usize, const NC: usize>(
        src: &<Self as HasStorage<T, { R * C }>>::Storage,
        dst: &mut <Self as HasStorage<T, { NR * NC }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }> + HasStorage<T, { NR * NC }>,
    {
        let s = <Self as HasStorage<T, { R * C }>>::as_slice(src);
        let d = <Self as HasStorage<T, { NR * NC }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }

    fn reshape32<const D0: usize, const D1: usize, const D2: usize, const R: usize, const C: usize>(
        src: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { R * C }>,
    {
        let s = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(src);
        let d = <Self as HasStorage<T, { R * C }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }

    fn reshape33<const D0: usize, const D1: usize, const D2: usize, const ND0: usize, const ND1: usize, const ND2: usize>(
        src: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { ND0 * (ND1 * ND2) }>,
    {
        let s = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(src);
        let d = <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }

    fn reshape44<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const ND0: usize,
        const ND1: usize,
        const ND2: usize,
        const ND3: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>,
    {
        let s = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(src);
        let d = <Self as HasStorage<T, { ND0 * (ND1 * (ND2 * ND3)) }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }

    fn reshape43<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const ND0: usize,
        const ND1: usize,
        const ND2: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { ND0 * (ND1 * ND2) }>,
    {
        let s = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(src);
        let d = <Self as HasStorage<T, { ND0 * (ND1 * ND2) }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }

    fn reshape42<
        const D0: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const R: usize,
        const C: usize,
    >(
        src: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        dst: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>
            + HasStorage<T, { R * C }>,
    {
        let s = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(src);
        let d = <Self as HasStorage<T, { R * C }>>::as_mut_slice(dst);
        d.copy_from_slice(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor2, Tensor3, Tensor4};

    #[test]
    fn test_reshape22() {
        let t = Tensor2::<i32, 2, 3, NaiveCpu>::new([1, 2, 3, 4, 5, 6]);
        let r = t.reshape2::<3, 2>();
        assert_eq!(r.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reshape32() {
        let t = Tensor3::<i32, 1, 2, 3, NaiveCpu>::new([1, 2, 3, 4, 5, 6]);
        let r = t.reshape2::<3, 2>();
        assert_eq!(r.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reshape33() {
        let t = Tensor3::<i32, 1, 2, 3, NaiveCpu>::new([1, 2, 3, 4, 5, 6]);
        let r = t.reshape3::<3, 2, 1>();
        assert_eq!(r.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reshape44() {
        let t = Tensor4::<i32, 1, 2, 3, 4, NaiveCpu>::new([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
        let r = t.reshape4::<2, 3, 4, 1>();
        assert_eq!(r.as_slice(), &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
    }

    #[test]
    fn test_reshape43() {
        let t = Tensor4::<i32, 1, 2, 3, 4, NaiveCpu>::new([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
        let r = t.reshape3::<4, 3, 2>();
        assert_eq!(r.as_slice(), &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
    }

    #[test]
    fn test_reshape42() {
        let t = Tensor4::<i32, 1, 2, 3, 4, NaiveCpu>::new([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
        let r = t.reshape2::<6, 4>();
        assert_eq!(r.as_slice(), &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
    }
}
