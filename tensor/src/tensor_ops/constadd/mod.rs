pub mod cast;  
pub mod naive_cpu; 

pub use cast::CastInto;
use core::ops::Add;
use crate::tensor::Tensor2;
use crate::storage::HasStorage;

pub trait ConstAdd<T: Copy + Default>: Sized {
    fn constadd<const R: usize, const C: usize>(
        a:   &<Self as HasStorage<T, { R * C }>>::Storage,
        k:   T,
        out: &mut <Self as HasStorage<T, { R * C }>>::Storage,
    )
    where
        T: Add<Output = T>,
        Self: HasStorage<T, { R * C }>;
}

//tensor2 + scalar
impl<T, S, const R: usize, const C: usize, B> Add<S> for Tensor2<T, R, C, B>
where
    T: Copy + Default + Add<Output = T>,
    S: Copy + CastInto<T>,
    B: ConstAdd<T> + HasStorage<T, { R * C }>,
{
    type Output = Tensor2<T, R, C, B>;

    #[inline]
    fn add(self, rhs: S) -> Self::Output {
        let k: T = rhs.cast();
        let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
        B::constadd::<R, C>(&self.storage, k, &mut out);
        Self::Output { storage: out, _p: core::marker::PhantomData }
    }
}

//scalar + tensor2
macro_rules! impl_scalar_plus_tensor {
    ($($Scalar:ty),* $(,)?) => {$(
        impl<T, const R: usize, const C: usize, B>
            Add<Tensor2<T, R, C, B>> for $Scalar
        where
            T: Copy + Default + Add<Output = T>,
            B: ConstAdd<T> + HasStorage<T, { R * C }>,
            $Scalar: CastInto<T> + Copy,
        {
            type Output = Tensor2<T, R, C, B>;

            #[inline]
            fn add(self, rhs: Tensor2<T, R, C, B>) -> Self::Output {
                /* convert the scalar, launch backend once */
                let k: T = self.cast();
                let mut out = <B as HasStorage<T, { R * C }>>::storage_uninit();
                B::constadd::<R, C>(&rhs.storage, k, &mut out);
                Tensor2 { storage: out, _p: core::marker::PhantomData }
            }
        }
    )*};
}

impl_scalar_plus_tensor!(i8, i16, i32, u8, u16, u32, f32, f64);
