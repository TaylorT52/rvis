use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::reduce::{Mean3, Sum3};
use core::ops::Add;
use num_traits::Float;

impl<T> Sum3<T> for NaiveCpu
where
    T: Copy + Default + Add<Output = T>,
{
    fn sum_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D1 * D2 }>>::as_mut_slice(out);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                let mut acc = T::default();
                for i0 in 0..D0 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i1 * D2 + i2] = acc;
            }
        }
    }

    fn sum_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D2 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i2 in 0..D2 {
                let mut acc = T::default();
                for i1 in 0..D1 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i0 * D2 + i2] = acc;
            }
        }
    }

    fn sum_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D1 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut acc = T::default();
                for i2 in 0..D2 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i0 * D1 + i1] = acc;
            }
        }
    }
}

impl<T> Mean3<T> for NaiveCpu
where
    T: Float + Default,
{
    fn mean_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D1 * D2 }>>::as_mut_slice(out);
        let denom = T::from(D0 as u32).unwrap();
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                let mut acc = T::zero();
                for i0 in 0..D0 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i1 * D2 + i2] = acc / denom;
            }
        }
    }

    fn mean_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D2 }>>::as_mut_slice(out);
        let denom = T::from(D1 as u32).unwrap();
        for i0 in 0..D0 {
            for i2 in 0..D2 {
                let mut acc = T::zero();
                for i1 in 0..D1 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i0 * D2 + i2] = acc / denom;
            }
        }
    }

    fn mean_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D1 }>>::as_mut_slice(out);
        let denom = T::from(D2 as u32).unwrap();
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut acc = T::zero();
                for i2 in 0..D2 {
                    acc = acc + src[i0 * D1 * D2 + i1 * D2 + i2];
                }
                dst[i0 * D1 + i1] = acc / denom;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor3;

    #[test]
    fn test_sum_axes() {
        let t = Tensor3::<f32, 2, 2, 2, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let s0 = t.sum_axis0();
        assert_eq!(s0.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

        let s1 = t.sum_axis1();
        assert_eq!(s1.as_slice(), &[4.0, 6.0, 12.0, 14.0]);

        let s2 = t.sum_axis2();
        assert_eq!(s2.as_slice(), &[3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn test_mean_axes() {
        let t = Tensor3::<f32, 2, 2, 2, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let m0 = t.mean_axis0();
        let m0s = m0.as_slice();
        assert!((m0s[0] - 3.0).abs() < 1e-6);
        assert!((m0s[1] - 4.0).abs() < 1e-6);
        assert!((m0s[2] - 5.0).abs() < 1e-6);
        assert!((m0s[3] - 6.0).abs() < 1e-6);

        let m1 = t.mean_axis1();
        assert_eq!(m1.as_slice(), &[2.0, 3.0, 6.0, 7.0]);

        let m2 = t.mean_axis2();
        assert_eq!(m2.as_slice(), &[1.5, 3.5, 5.5, 7.5]);
    }
}
