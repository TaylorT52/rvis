use crate::storage::HasStorage;
use crate::storage::naive_cpu::NaiveCpu;
use crate::tensor_ops::slicing::Slice;

impl<T> Slice<T> for NaiveCpu
where
    T: Copy + Default,
{
    fn slice2<const R: usize, const C: usize, const R_START: usize, const R_END: usize, const C_START: usize, const C_END: usize>(
        a: &<Self as HasStorage<T, { R * C }>>::Storage,
        out: &mut <Self as HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>>::Storage,
    ) where
        Self: HasStorage<T, { R * C }> + HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>,
    {
        let src = <Self as HasStorage<T, { R * C }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { (R_END - R_START) * (C_END - C_START) }>>::as_mut_slice(out);
        
        for r in R_START..R_END {
            for c in C_START..C_END {
                let src_idx = r * C + c;
                let dst_idx = (r - R_START) * (C_END - C_START) + (c - C_START);
                dst[dst_idx] = src[src_idx];
            }
        }
    }

    fn slice3<const D0: usize, const D1: usize, const D2: usize, const DIM: usize, const START: usize, const END: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * D2) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * D2) }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * (D1 * D2) }>>::as_mut_slice(out);
        
        match DIM {
            0 => {
                for d0 in START..END {
                    for d1 in 0..D1 {
                        for d2 in 0..D2 {
                            let src_idx = d0 * (D1 * D2) + d1 * D2 + d2;
                            let dst_idx = (d0 - START) * (D1 * D2) + d1 * D2 + d2;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
            1 => {
                for d0 in 0..D0 {
                    for d1 in START..END {
                        for d2 in 0..D2 {
                            let src_idx = d0 * (D1 * D2) + d1 * D2 + d2;
                            let dst_idx = d0 * ((END - START) * D2) + (d1 - START) * D2 + d2;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
            2 => {
                for d0 in 0..D0 {
                    for d1 in 0..D1 {
                        for d2 in START..END {
                            let src_idx = d0 * (D1 * D2) + d1 * D2 + d2;
                            let dst_idx = d0 * (D1 * (END - START)) + d1 * (END - START) + (d2 - START);
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
            _ => panic!("Invalid dimension for 3D tensor slicing"),
        }
    }

    fn slice4<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const DIM: usize, const START: usize, const END: usize>(
        a: &<Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
        out: &mut <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::Storage,
    ) where
        Self: HasStorage<T, { D0 * (D1 * (D2 * D3)) }>,
    {
        let src = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_slice(a);
        let dst = <Self as HasStorage<T, { D0 * (D1 * (D2 * D3)) }>>::as_mut_slice(out);
        
        match DIM {
            0 => {
                for d0 in START..END {
                    for d1 in 0..D1 {
                        for d2 in 0..D2 {
                            for d3 in 0..D3 {
                                let src_idx = d0 * (D1 * (D2 * D3)) + d1 * (D2 * D3) + d2 * D3 + d3;
                                let dst_idx = (d0 - START) * (D1 * (D2 * D3)) + d1 * (D2 * D3) + d2 * D3 + d3;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
            1 => {
                for d0 in 0..D0 {
                    for d1 in START..END {
                        for d2 in 0..D2 {
                            for d3 in 0..D3 {
                                let src_idx = d0 * (D1 * (D2 * D3)) + d1 * (D2 * D3) + d2 * D3 + d3;
                                let dst_idx = d0 * ((END - START) * (D2 * D3)) + (d1 - START) * (D2 * D3) + d2 * D3 + d3;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
            2 => {
                for d0 in 0..D0 {
                    for d1 in 0..D1 {
                        for d2 in START..END {
                            for d3 in 0..D3 {
                                let src_idx = d0 * (D1 * (D2 * D3)) + d1 * (D2 * D3) + d2 * D3 + d3;
                                let dst_idx = d0 * (D1 * ((END - START) * D3)) + d1 * ((END - START) * D3) + (d2 - START) * D3 + d3;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
            3 => {
                for d0 in 0..D0 {
                    for d1 in 0..D1 {
                        for d2 in 0..D2 {
                            for d3 in START..END {
                                let src_idx = d0 * (D1 * (D2 * D3)) + d1 * (D2 * D3) + d2 * D3 + d3;
                                let dst_idx = d0 * (D1 * (D2 * (END - START))) + d1 * (D2 * (END - START)) + d2 * (END - START) + (d3 - START);
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
            _ => panic!("Invalid dimension for 4D tensor slicing"),
        }
    }
}
