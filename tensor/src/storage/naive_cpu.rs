use crate::storage::HasStorage;

pub struct NaiveCpu;

// Generic implementation for *any* size N.
impl<T: Copy + Default, const N: usize> HasStorage<T, N> for NaiveCpu {
    type Storage = [T; N];

    #[inline]
    fn storage_from_array(src: [T; N]) -> Self::Storage {
        src
    }

    #[inline]
    fn storage_uninit() -> Self::Storage {
        [T::default(); N]
    }

    #[inline]
    fn storage_zeroes() -> Self::Storage {
        [T::default(); N]
    }

    #[inline]
    fn storage_ones() -> Self::Storage
    where
        T: num_traits::One,
    {
        [T::one(); N]
    }

    #[inline]
    fn storage_full(val: T) -> Self::Storage {
        [val; N]
    }

    #[inline]
    fn as_slice(storage: &Self::Storage) -> &[T] {
        storage
    }

    #[inline]
    fn as_mut_slice(storage: &mut Self::Storage) -> &mut [T] {
        storage
    }
}
