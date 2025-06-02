use std::marker::PhantomData;
use crate::backend::Backend;
use crate::tensor::helper::flatten_idx;

mod helper;

#[derive(Clone, Debug)]
pub struct StaticTensor<T, const D: usize, const SHAPE: [usize; D], B: Backend<T>>
where
    T: Copy + Default,
{
    storage: B::Storage,
    _p:      PhantomData<T>,
}

impl<T, const D: usize, const SHAPE: [usize; D], B> StaticTensor<T, D, SHAPE, B>
where
    T: Copy + Default,
    B: Backend<T>,
{
    pub const RANK: usize = D;

    pub const fn elem_count() -> usize {
        let mut n = 1;
        let mut i = 0;
        while i < D { n *= SHAPE[i]; i += 1; }
        n
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        assert_eq!(data.len(), Self::elem_count());
        Self { storage: B::storage_from_vec(data), _p: PhantomData }
    }

    pub const fn shape() -> [usize; D] { SHAPE }

    pub fn to_vec(&self) -> Vec<T> { B::as_slice(&self.storage).to_vec() }
}

const fn bdim(a: usize, b: usize) -> usize {
    if a == 1 { b } else if b == 1 { a } else if a == b { a } else {
        // format macros not supported in const fn
        panic!("dimension mismatch in broadcast");
    }
}
const fn broadcast<const D: usize>(a: [usize; D], b: [usize; D]) -> [usize; D] {
    let mut out = [0; D];
    let mut i = 0;
    while i < D { out[i] = bdim(a[i], b[i]); i += 1; }
    out
}

macro_rules! impl_static_binop {
    ($trait:ident, $func:ident) => {
        impl<T, const D: usize, const A: [usize; D], const B_: [usize; D], BK>
            core::ops::$trait<StaticTensor<T, D, B_, BK>>
            for StaticTensor<T, D, A, BK>
        where
            T: Copy + Default + core::ops::$trait<Output = T>,
            BK: Backend<T>,
        {
            type Output = StaticTensor<T, D, { broadcast(A, B_) }, BK>;

            fn $func(self, rhs: StaticTensor<T, D, B_, BK>) -> Self::Output {
                let out_shape = broadcast(A, B_);
                let out_len   = out_shape.iter().product();
                let mut out   = BK::storage_uninit(out_len);
                BK::$func(&self.storage, &rhs.storage, &mut out);
                StaticTensor { storage: out, _p: PhantomData }
            }
        }
    };
}
impl_static_binop!(Add, add);
impl_static_binop!(Sub, sub);
impl_static_binop!(Mul, mul);
impl_static_binop!(Div, div);

#[derive(Clone, Debug)]
pub struct DynTensor<T, const D: usize, B: Backend<T>>
where
    T: Copy + Default,
{
    shape:   [usize; D],
    storage: B::Storage,
}

impl<T, const D: usize, B> DynTensor<T, D, B>
where
    T: Copy + Default,
    B: Backend<T>,
{
    pub const fn rank() -> usize { D }

    pub fn new(shape: [usize; D], data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, storage: B::storage_from_vec(data) }
    }

    pub fn elem_count(&self) -> usize { self.shape.iter().product() }

    pub fn shape(&self) -> &[usize; D] { &self.shape }

    pub fn to_vec(&self) -> Vec<T> { B::as_slice(&self.storage).to_vec() }
}

macro_rules! impl_dyn_static_binop {
    ($trait:ident, $func:ident) => {
        impl<T, const D: usize, const SH: [usize; D], BK>
            core::ops::$trait<StaticTensor<T, D, SH, BK>> for DynTensor<T, D, BK>
        where
            T: Copy + Default + core::ops::$trait<Output = T>,
            BK: Backend<T>,
        {
            type Output = DynTensor<T, D, BK>;
            fn $func(self, rhs: StaticTensor<T, D, SH, BK>) -> Self::Output {
                let out_shape = broadcast(self.shape, SH);
                let out_len   = out_shape.iter().product();
                let mut out   = BK::storage_uninit(out_len);
                BK::$func(&self.storage, &rhs.storage, &mut out);
                DynTensor { shape: out_shape, storage: out }
            }
        }

        impl<T, const D: usize, const SH: [usize; D], BK>
            core::ops::$trait<DynTensor<T, D, BK>> for StaticTensor<T, D, SH, BK>
        where
            T: Copy + Default + core::ops::$trait<Output = T>,
            BK: Backend<T>,
        {
            type Output = DynTensor<T, D, BK>;
            fn $func(self, rhs: DynTensor<T, D, BK>) -> Self::Output {
                rhs.$func(self)
            }
        }
    };
}
impl_dyn_static_binop!(Add, add);
impl_dyn_static_binop!(Sub, sub);
impl_dyn_static_binop!(Mul, mul);
impl_dyn_static_binop!(Div, div);

#[doc(hidden)]
#[macro_export]
macro_rules! __count {
    () => {0usize};
    ($_head:expr $(, $tail:expr)*) => {1usize + $crate::__count!($($tail),*)};
}


/// **`stensor!`** – create a *static‑shape* tensor literal quickly.
///
/// ```rust
/// use tensor::statt;
/// let a = stat!([1, 2, 3, 4]; [2, 2]); // rank‑2 matrix 2×2 on CPU
/// ```
#[macro_export]
macro_rules! statt {
    ([$($data:expr),* $(,)?] ; [$($dim:expr),* $(,)?]) => {{
        const SHAPE: [usize; $crate::__count!($($dim),*)] = [ $($dim),* ];
        $crate::StaticTensor::<_, { $crate::__count!($($dim),*) }, SHAPE>::from_vec(vec![ $($data),* ])
    }};
}

/// **`dtensor!`** – create a *dynamic* tensor literal.
/// ```rust
/// use tensor::dynt;
/// let b = dynt!([1, 2, 3, 4]; [2, 2]);
/// ```
#[macro_export]
macro_rules! dynt {
    ([$($data:expr),* $(,)?] ; [$($dim:expr),* $(,)?]) => {{
        const RANK: usize = $crate::__count!($($dim),*);
        $crate::DynTensor::<_, RANK>::new([ $($dim),* ], vec![ $($data),* ])
    }};
}