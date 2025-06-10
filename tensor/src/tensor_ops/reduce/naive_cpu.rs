use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::reduce::{Argmax3, Argmax4, Max3, Max4, Mean3, Mean4, Sum3};
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

impl<T> Max3<T> for NaiveCpu
where
    T: Copy + Default + PartialOrd,
{
    fn max_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D1 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D1 * D2 }>>::as_mut_slice(out);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                let mut best = src[i1 * D2 + i2];
                for i0 in 1..D0 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                    }
                }
                dst[i1 * D2 + i2] = best;
            }
        }
    }

    fn max_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D2 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i2 in 0..D2 {
                let mut best = src[i0 * D1 * D2 + i2];
                for i1 in 1..D1 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                    }
                }
                dst[i0 * D2 + i2] = best;
            }
        }
    }

    fn max_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<T, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D1 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut best = src[i0 * D1 * D2 + i1 * D2];
                for i2 in 1..D2 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                    }
                }
                dst[i0 * D1 + i1] = best;
            }
        }
    }
}

impl<T> Argmax3<T> for NaiveCpu
where
    T: Copy + Default + PartialOrd,
{
    fn argmax_axis0<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D1 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D1 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<usize, { D1 * D2 }>>::as_mut_slice(out);
        for i1 in 0..D1 {
            for i2 in 0..D2 {
                let mut best = src[i1 * D2 + i2];
                let mut idx = 0usize;
                for i0 in 1..D0 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                        idx = i0;
                    }
                }
                dst[i1 * D2 + i2] = idx;
            }
        }
    }

    fn argmax_axis1<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D2 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D0 * D2 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<usize, { D0 * D2 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i2 in 0..D2 {
                let mut best = src[i0 * D1 * D2 + i2];
                let mut idx = 0usize;
                for i1 in 1..D1 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                        idx = i1;
                    }
                }
                dst[i0 * D2 + i2] = idx;
            }
        }
    }

    fn argmax_axis2<const D0: usize, const D1: usize, const D2: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }> + HasStorage<usize, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<usize, { D0 * D1 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut best = src[i0 * D1 * D2 + i1 * D2];
                let mut idx = 0usize;
                for i2 in 1..D2 {
                    let val = src[i0 * D1 * D2 + i1 * D2 + i2];
                    if val > best {
                        best = val;
                        idx = i2;
                    }
                }
                dst[i0 * D1 + i1] = idx;
            }
        }
    }
}

impl<T> Mean4<T> for NaiveCpu
where
    T: Float + Default,
{
    fn mean_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D1 }>>::as_mut_slice(out);
        let denom = T::from((D2 * D3) as u32).unwrap();
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut acc = T::zero();
                for i2 in 0..D2 {
                    for i3 in 0..D3 {
                        acc = acc + src[((i0 * D1 + i1) * D2 + i2) * D3 + i3];
                    }
                }
                dst[i0 * D1 + i1] = acc / denom;
            }
        }
    }
}

impl<T> Max4<T> for NaiveCpu
where
    T: Copy + Default + PartialOrd,
{
    fn max_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<T, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * D1 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut best = src[((i0 * D1 + i1) * D2) * D3];
                for i2 in 0..D2 {
                    for i3 in 0..D3 {
                        let val = src[((i0 * D1 + i1) * D2 + i2) * D3 + i3];
                        if val > best {
                            best = val;
                        }
                    }
                }
                dst[i0 * D1 + i1] = best;
            }
        }
    }
}

impl<T> Argmax4<T> for NaiveCpu
where
    T: Copy + Default + PartialOrd,
{
    fn argmax_axis23<const D0: usize, const D1: usize, const D2: usize, const D3: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<usize, { D0 * D1 }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }> + HasStorage<usize, { D0 * D1 }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(a);
        let dst = <Self as HasStorage<usize, { D0 * D1 }>>::as_mut_slice(out);
        for i0 in 0..D0 {
            for i1 in 0..D1 {
                let mut best = src[((i0 * D1 + i1) * D2) * D3];
                let mut idx = 0usize;
                for i2 in 0..D2 {
                    for i3 in 0..D3 {
                        let val = src[((i0 * D1 + i1) * D2 + i2) * D3 + i3];
                        let flat = i2 * D3 + i3;
                        if val > best {
                            best = val;
                            idx = flat;
                        }
                    }
                }
                dst[i0 * D1 + i1] = idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor3, Tensor4};

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

    #[test]
    fn test_max_and_argmax_axes() {
        let t = Tensor3::<f32, 2, 2, 2, NaiveCpu>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mx0 = t.max_axis0();
        assert_eq!(mx0.as_slice(), &[5.0, 6.0, 7.0, 8.0]);
        let am0 = t.argmax_axis0();
        assert_eq!(am0.as_slice(), &[1, 1, 1, 1]);

        let mx1 = t.max_axis1();
        assert_eq!(mx1.as_slice(), &[3.0, 4.0, 7.0, 8.0]);
        let am1 = t.argmax_axis1();
        assert_eq!(am1.as_slice(), &[1, 1, 1, 1]);

        let mx2 = t.max_axis2();
        assert_eq!(mx2.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
        let am2 = t.argmax_axis2();
        assert_eq!(am2.as_slice(), &[1, 1, 1, 1]);

        let mb = t.mean_batches();
        assert_eq!(mb.as_slice(), &[1.5, 3.5, 5.5, 7.5]);
        let xb = t.max_batches();
        assert_eq!(xb.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
        let ab = t.argmax_batches();
        assert_eq!(ab.as_slice(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_reduce_tensor4_batches() {
        let t = Tensor4::<f32, 2, 2, 2, 2, NaiveCpu>::new([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);

        let m = t.mean_axis23();
        assert_eq!(m.as_slice(), &[3.5, 7.5, 11.5, 15.5]);

        let mx = t.max_axis23();
        assert_eq!(mx.as_slice(), &[4.0, 8.0, 12.0, 16.0]);

        let am = t.argmax_axis23();
        assert_eq!(am.as_slice(), &[3, 3, 3, 3]);
    }
}
