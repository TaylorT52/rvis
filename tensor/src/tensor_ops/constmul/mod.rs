pub mod cast;
pub mod naive_cpu;

use core::ops::Mul;
use crate::tensor::Tensor2;
use crate::storage::HasStorage;
use cast::CastInto;

pub trait ConstMul<T: Copy + Default>: Sized {
    fn constmul<const R: usize, const C: usize>(
        a:   &<Self as HasStorage<T, { R * C }>>::Storage,
        k:   T,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    )
    where
        T: Mul<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

//scalar * tensor2
impl<T, S, const R: usize, const C: usize, B> Mul<S> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Mul<Output = T>,
    S: Copy + CastInto<T>,
    B: ConstMul<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn mul(self, rhs: S) -> Self::Output {
        let k: T = rhs.cast();
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::constmul::<R, C>(&self.storage, k, &mut out);
        Self::Output { storage: out, _p: core::marker::PhantomData }
    }
}

//tensor2 * scalar
macro_rules! impl_scalar_times_tensor {
    ($($Scalar:ty),* $(,)?) => {$(
        impl<T, const R: usize, const C: usize, B>
            core::ops::Mul<Tensor2<T, R, C, B>> for $Scalar
        where
            T: Copy + Default + core::ops::Mul<Output = T>,
            B: ConstMul<T> + HasStorage<T, { R * C }>,
            $Scalar: CastInto<T> + Copy,
        {
            type Output = Tensor2<T, R, C, B>;

            #[inline]
            fn mul(self, rhs: Tensor2<T, R, C, B>) -> Self::Output {
                let k: T = self.cast();
                let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
                B::constmul::<R, C>(&rhs.storage, k, &mut out);
                Tensor2 { storage: out, _p: core::marker::PhantomData }
            }
        }
    )*};
}

impl_scalar_times_tensor!(i8, i16, i32, u8, u16, u32, f32, f64);
