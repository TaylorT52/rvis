pub fn flatten_idx<const D: usize>(idx: &[usize; D], shape: &[usize; D]) -> usize {
    let mut stride = 1;
    let mut offset = 0;
    let mut d = D as isize - 1;
    while d >= 0 {
        offset += idx[d as usize] * stride;
        stride *= shape[d as usize];
        d -= 1;
    }
    offset
}
