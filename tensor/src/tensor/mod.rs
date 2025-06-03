use crate::backend::Backend;
use core::ops::Mul;
use std::marker::PhantomData;

pub trait StaticShape {
    const D:     usize;
    const SHAPE: &'static [usize];

    #[inline(always)]
    fn shape(&self) -> [usize; Self::D] {
        let mut out = [0; Self::D];
        out.copy_from_slice(Self::SHAPE);
        out
    }

    #[inline(always)]
    fn size(&self) -> usize {
        Self::SHAPE.iter().copied().product()
    }

    // TODO: strides
}

macro_rules! impl_tensor_rank {
    ($name:ident, [$($dim:ident),+]) => {
        pub struct $name<T, $(const $dim: usize),+, B: Backend<T>>
        where
            T: Copy + Default,
        {
            storage: B::Storage,
            _p: PhantomData<T>,
        }

        impl<T, $(const $dim: usize),+, B: Backend<T>> StaticShape for $name<T, $($dim),+, B>
        where
            T: Copy + Default,
        {
            const D: usize = impl_tensor_rank!(@count_dims $($dim),*);
            const SHAPE: &'static [usize] = &[ $($dim),* ];
        }

        impl<T, $(const $dim: usize),+, B: Backend<T>> $name<T, $($dim),+, B>
        where
            T: Copy + Default,
        {
            pub fn new(data: [T; impl_tensor_rank!(@product $($dim),*)]) -> Self {
                Self {
                    storage: B::storage_from_vec(data.to_vec()),
                    _p: PhantomData,
                }
            }
        }
    };

    (@count_dims $head:ident $(, $tail:ident)*) => {
        1 $(+ { let _ = $tail; 1 })*
    };

    (@product $head:ident $(, $tail:ident)*) => {
        $head $( * $tail )*
    };
}

impl_tensor_rank!(Tensor2, [R, C]);
impl_tensor_rank!(Tensor3, [D0, D1, D2]);
impl_tensor_rank!(Tensor4, [D0, D1, D3, D4]);


/// 2-dimensional matrix multiplication
impl<T, const R: usize, const C: usize, const K: usize, B> Mul<Tensor2<T, C, K, B>>
for Tensor2<T, R, C, B>
where
    T: Copy + Default + Mul<Output = T>,
    B: Backend<T>,
{
    type Output = Tensor2<T, R, K, B>;

    fn mul(self, rhs: Tensor2<T, C, K, B>) -> Self::Output {
        let mut b = B::storage_uninit(R * K);
        B::matmul::<R, C, K>(&self.storage, &rhs.storage, &mut b);
        Self::Output {
            storage: b,
            _p: PhantomData,
        }
    }
}

// TODO: indexing
// for indexing of 1, N
// TODO: add/mul/sub/div for same shape
// TODO: multiply/add/div/sub by constant
// TODO: reshaping
// TODO: transpose