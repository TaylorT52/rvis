//need this for type issues with const mul
pub trait CastInto<T> { fn cast(self) -> T; }
impl<T> CastInto<T> for T { #[inline] fn cast(self) -> T { self } }
macro_rules! impl_cast { ($s:ty => $d:ty) => {
    impl CastInto<$d> for $s { #[inline] fn cast(self) -> $d { self as $d } }
}; }
impl_cast!(i8  => f32); impl_cast!(i16 => f32); impl_cast!(i32 => f32);
impl_cast!(u8  => f32); impl_cast!(u16 => f32); impl_cast!(u32 => f32);
impl_cast!(f32 => f64); impl_cast!(i32 => f64);
