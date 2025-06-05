pub mod metal_gpu;
pub mod naive_cpu;

pub trait HasStorage<T: Copy + Default, const N: usize> {
    type Storage;

    fn storage_from_array(src: [T; N]) -> Self::Storage;
    fn storage_uninit() -> Self::Storage;
    fn as_slice(storage: &Self::Storage) -> &[T];
    fn as_mut_slice(storage: &mut Self::Storage) -> &mut [T];
}
