use crate::storage::HasStorage;
use std::fmt;
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
            S,
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

            pub fn new_from_slice(data: &[T]) -> Self {
                Self {
                    storage: S::storage_from_array(data.try_into().unwrap()),
                    _p: PhantomData,
                }
            }

            #[inline]
            pub fn zeroes() -> Self {
                Self {
                    storage: S::storage_zeroes(),
                    _p: PhantomData,
                }
            }

            #[inline]
            pub fn ones() -> Self
            where
                T: num_traits::One,
            {
                Self {
                    storage: S::storage_ones(),
                    _p: PhantomData,
                }
            }

            #[inline]
            pub fn full(val: T) -> Self {
                Self {
                    storage: S::storage_full(val),
                    _p: PhantomData,
                }
            }

            #[inline]
            pub fn as_slice(&self) -> &[T] {
                S::as_slice(&self.storage)
            }
        }

        // ——— COPY + CLONE for each tensor rank ———
        impl<T, $(const $dim: usize,)+ S> Copy for $name<T, $($dim,)+ S>
        where
            T: Copy + Default,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
            // the storage must be Copy
            S::Storage: Copy,
        {}

        impl<T, $(const $dim: usize,)+ S> Clone for $name<T, $($dim,)+ S>
        where
            T: Copy + Default,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
            // the storage must be Clone
            S::Storage: Clone,
        {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    storage: self.storage.clone(),
                    _p: PhantomData,
                }
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

/// Helper for rendering N-dimensional tensors with indentation.
fn fmt_nd<T: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
    depth: usize,
) -> fmt::Result {
    if depth + 1 == shape.len() {
        write!(f, "[")?;
        for i in 0..shape[depth] {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", data[i])?;
        }
        return write!(f, "]");
    }

    let chunk: usize = shape[depth + 1..].iter().product();
    write!(f, "[")?;
    for i in 0..shape[depth] {
        if i > 0 {
            write!(f, ",\n")?;
            for _ in 0..=depth {
                write!(f, " ")?;
            }
        }
        fmt_nd(f, &data[i * chunk..(i + 1) * chunk], shape, depth + 1)?;
    }
    write!(f, "]")
}

macro_rules! impl_tensor_display {
    ($name:ident, [$($dim:ident),+]) => {
        impl<T, $(const $dim: usize,)+ S> fmt::Display for $name<T, $($dim,)+ S>
        where
            T: Copy + Default + fmt::Display,
            S: HasStorage<T, { impl_tensor_rank!(@prod $($dim),+) }>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_nd(f, self.as_slice(), &[ $($dim),+ ], 0)
            }
        }
    };
}

impl_tensor_display!(Tensor2, [R, C]);
impl_tensor_display!(Tensor3, [D0, D1, D2]);
impl_tensor_display!(Tensor4, [D0, D1, D3, D4]);

// TODO: reshaping
// TODO: transpose
