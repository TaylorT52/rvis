//! Minimal numeric‑conversion helpers so `tensor + 5` and `5 + tensor` work
//! when the tensor stores `f32` (or other supported `T`).

pub trait CastInto<T> { fn cast(self) -> T; }

impl<T> CastInto<T> for T { #[inline] fn cast(self) -> T { self } }

macro_rules! impl_cast { ($s:ty => $d:ty) => {
    impl CastInto<$d> for $s { #[inline] fn cast(self) -> $d { self as $d } }
}; }

/* ints → floats */
impl_cast!(i8  => f32);   impl_cast!(i8  => f64);
impl_cast!(i16 => f32);   impl_cast!(i16  => f64);
impl_cast!(i32 => f32);   impl_cast!(i32  => f64);   // ← MISSING LINE
impl_cast!(u8  => f32);   impl_cast!(u8   => f64);
impl_cast!(u16 => f32);   impl_cast!(u16  => f64);
impl_cast!(u32 => f32);   impl_cast!(u32  => f64);
/* float ↔ float */
impl_cast!(f32 => f64);   impl_cast!(f64  => f32);

