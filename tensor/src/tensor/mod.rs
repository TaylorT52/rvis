use crate::storage::HasStorage;
use std::marker::PhantomData;

pub trait StaticShape {
    const D: usize;
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
}

macro_rules! impl_tensor_rank {
    ($name:ident, [$($dim:ident),+]) => {
        pub struct $name<
            T,
            $(const $dim: usize,)+
            S,          // default back‑end
        >
        where
            T: Copy + Default,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
        {
            pub(crate) storage:
                <S as HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>>::Storage,
            pub(crate) _p: PhantomData<T>,
        }

        impl<T, $(const $dim: usize,)+ S> StaticShape for $name<T, $($dim,)+ S>
        where
            T: Copy + Default,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
        {
            const D: usize = impl_tensor_rank!(@count $($dim),+);
            const SHAPE: &'static [usize] = &[ $($dim),+ ];
        }

        /* --- convenience constructors & access --- */
        impl<T, $(const $dim: usize,)+ S> $name<T, $($dim,)+ S>
        where
            T: Copy + Default,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
        {
            pub fn new(data: [T; impl_tensor_rank!(@prod $($dim),+)]) -> Self {
                Self {
                    storage: S::storage_from_array(data),
                    _p: PhantomData,
                }
            }

            #[inline]
            pub fn as_slice(&self) -> &[T] {
                S::as_slice(&self.storage)
            }
        }
    };

    /* helper rules */
    (@count $first:ident $(,$rest:ident)+) => { 1 + impl_tensor_rank!(@count $($rest),+) };
    (@count $only:ident)                   => { 1 };
    (@prod  $first:ident $(,$rest:ident)+) => { $first * impl_tensor_rank!(@prod $($rest),+) };
    (@prod  $only:ident)                   => { $only };
}

impl_tensor_rank!(Tensor2, [R, C]);
impl_tensor_rank!(Tensor3, [D0, D1, D2]);
impl_tensor_rank!(Tensor4, [D0, D1, D3, D4]);

// TODO: indexing
// for indexing of 1, N
// TODO: add/mul/sub/div for same shape
// TODO: multiply/add/div/sub by constant
// TODO: reshaping
// TODO: transpose
