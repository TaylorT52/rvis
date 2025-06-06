use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::relu::Relu;
use core::cmp::PartialOrd;

impl<T> Relu<T> for NaiveCpu
where
    T: Copy + Default + PartialOrd,
{
    fn relu<const N: usize>(
        a: &<Self as HasStorage<T, N>>::Storage,
        out: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let src = <Self as HasStorage<T, N>>::as_slice(a);
        let dst = <Self as HasStorage<T, N>>::as_mut_slice(out);
        let zero = T::default();
        for i in 0..N {
            let v = src[i];
            dst[i] = if v > zero { v } else { zero };
        }
    }

    fn relu_backward<const N: usize>(
        input: &<Self as HasStorage<T, N>>::Storage,
        grad_output: &<Self as HasStorage<T, N>>::Storage,
        grad_input: &mut <Self as HasStorage<T, N>>::Storage,
    ) where
        Self: HasStorage<T, N>,
    {
        let input_data = <Self as HasStorage<T, N>>::as_slice(input);
        let grad_out = <Self as HasStorage<T, N>>::as_slice(grad_output);
        let grad_in = <Self as HasStorage<T, N>>::as_mut_slice(grad_input);
        let zero = T::default();

        for i in 0..N {
            grad_in[i] = if input_data[i] > zero { grad_out[i] } else { zero };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor2;

    #[test]
    fn test_relu_2x2() {
        let t = Tensor2::<f32, 2, 2, NaiveCpu>::new([-1.0, 0.5, -2.0, 3.0]);
        let out = t.relu();
        let s = out.as_slice();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 0.5);
        assert_eq!(s[2], 0.0);
        assert_eq!(s[3], 3.0);
    }
}
