use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::log::Log;
use num_traits::Float;

impl<T> Log<T> for NaiveCpu
where
    T: Copy + Default + Float,
{
    fn log<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        for i in 0..N {
            dst[i] = src[i].ln();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor2;

    #[test]
    fn test_log_2x2() {
        let t = Tensor2::<f32, 2, 2, NaiveCpu>::new([1.0, 2.7182817, 7.389056, 20.085537]);
        let out = t.log();
        let s = out.as_slice();
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);
        assert!((s[2] - 2.0).abs() < 1e-6);
        assert!((s[3] - 3.0).abs() < 1e-6);
    }
}
