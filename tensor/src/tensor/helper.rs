const fn prod<const N: usize>(a: &[usize; N]) -> usize {
    let mut p = 1;
    let mut i = 0;
    while i < N {
        p *= a[i];
        i += 1;
    }
    p
}

/// Are two shapes *exactly* equal?
const fn same_shape<const N: usize>(a: &[usize; N], b: &[usize; N]) -> bool {
    let mut i = 0;
    while i < N {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
