mod naive_cpu;

/// A back‑end is a thin façade that owns the hot loops.  Each tensor op calls
/// `B::add`, `B::mul`, … so swapping `NaiveCpu` for a BLAS or CUDA backend is
/// one `use` statement.
///
/// The trait is generic over the *scalar type* because different back‑ends may
/// have highly specialized kernels for `f32`, `f64`, `i32`, etc.
///
/// ✧ **Extension idea**: put the functions behind *associated* traits rather
///   than `T: Copy + Add`, to allow back‑ends to support *SIMD packed* types.
pub trait Backend<T: Copy + Default> {
    /// Concrete buffer that stores elements.
    type Storage;

    fn storage_from_vec(src: Vec<T>) -> Self::Storage;
    fn storage_uninit(len: usize) -> Self::Storage; // filled with `T::default()`
    fn as_slice(storage: &Self::Storage) -> &[T];
    fn as_mut_slice(storage: &mut Self::Storage) -> &mut [T];

    fn add(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage);
    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage);
    fn mul(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage);
    fn div(lhs: &Self::Storage, rhs: &Self::Storage, out: &mut Self::Storage);

    fn add_scalar(lhs: &Self::Storage, rhs: T, out: &mut Self::Storage);
    fn mul_scalar(lhs: &Self::Storage, rhs: T, out: &mut Self::Storage);
}
