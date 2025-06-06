use std::mem;
use crate::storage::HasStorage;
use crate::storage::metal_gpu::{MetalGpu, MetalGpuStorage};
use crate::tensor_ops::broadcast_matmul::BroadcastMatMul3;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

impl BroadcastMatMul3<f32> for MetalGpu {
    fn matmul3<const BATCH: usize, const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<f32, { BATCH * (R * C) }>>::Storage,
        b: &<Self as HasStorage<f32, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<f32, { BATCH * (R * K) }>>::Storage,
    ) where
        Self: HasStorage<f32, { BATCH * (R * C) }>
        + HasStorage<f32, { C * K }>
        + HasStorage<f32, { BATCH * (R * K) }>,
    {
        unsafe {
            // Get raw buffer handles
            let storage_a: &MetalGpuStorage = &*(a as *const _ as *const _);
            let storage_b: &MetalGpuStorage = &*(b as *const _ as *const _);
            let storage_out: &MetalGpuStorage = &*(out as *const _ as *const _);

            let device = &MetalGpu::shared().device;
            let queue = &MetalGpu::shared().queue;

            // Create one MPSMatrixMultiplication kernel (alpha = 1.0, beta = 0.0)
            let mm = {
                let mm = MPSMatrixMultiplication::alloc();
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    mm,
                    device,
                    false,   // transposeLeft = false
                    false,   // transposeRight = false
                    R as _,  // resultRows
                    K as _,  // resultColumns
                    C as _,  // interiorColumns
                    1.0,     // alpha
                    0.0,     // beta
                )
            };

            // Build a single “B” matrix view once (same for every batch)
            let element_size = mem::size_of::<f32>();
            let row_bytes_b = (K * element_size) as _; // bytes per row = K columns × 4 bytes
            let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                C as _,
                K as _,
                row_bytes_b,
                MPSDataType::Float32,
            );
            let mat_b = {
                let m = MPSMatrix::alloc();
                MPSMatrix::initWithBuffer_offset_descriptor(
                    m,
                    &storage_b.buffer,
                    0,         // offset in bytes into buffer
                    &desc_b,
                )
            };

            // Issue one command buffer and encode all batch‐slices into it
            let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                queue.commandBuffer().expect("expected a command buffer");

            for batch in 0..BATCH {
                // A: an R×C slice at offset (batch * R * C) elements
                let offset_a_bytes = (batch * R * C * element_size) as _;
                let row_bytes_a = (C * element_size) as _;
                let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    R as _,
                    C as _,
                    row_bytes_a,
                    MPSDataType::Float32,
                );
                let mat_a = {
                    let m = MPSMatrix::alloc();
                    MPSMatrix::initWithBuffer_offset_descriptor(
                        m,
                        &storage_a.buffer,
                        offset_a_bytes,
                        &desc_a,
                    )
                };

                // Out: an R×K slice at offset (batch * R * K) elements
                let offset_o_bytes = (batch * R * K * element_size) as _;
                let row_bytes_o = (K * element_size) as _;
                let desc_o = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    R as _,
                    K as _,
                    row_bytes_o,
                    MPSDataType::Float32,
                );
                let mat_out = {
                    let m = MPSMatrix::alloc();
                    MPSMatrix::initWithBuffer_offset_descriptor(
                        m,
                        &storage_out.buffer,
                        offset_o_bytes,
                        &desc_o,
                    )
                };

                // Encode one 2D matmul: (R×C) × (C×K) → (R×K)
                mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd_buf, &*mat_a, &*mat_b, &*mat_out,
                );
            }

            // Commit and wait until finished
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }
}

#[cfg(test)]
mod bench {
    use crate::storage::metal_gpu::MetalGpu;
    use crate::tensor::{Tensor2, Tensor3};
    use test::Bencher;

    #[bench]
    fn max_gpu_matmul(b: &mut Bencher) {
        const BATCHES: usize = 1023;
        const N: usize = 1024;

        // Allocate once — no cloning in hot loop
        let a = Tensor3::<f32, BATCHES, N, N, MetalGpu>::zeroes();
        let b_mat = Tensor2::<f32, N, N, MetalGpu>::zeroes();

        // Warmup to avoid first-run overheads
        let _ = a.clone() * b_mat.clone();

        b.iter(|| {
            // Perform one large batched matmul
            test::black_box(a.clone() * b_mat.clone());
        });
    }
}

