use crate::storage::HasStorage;
use crate::storage::metal_gpu::{MetalGpu, MetalGpuStorage};
use crate::tensor_ops::matmul::MatMul;
use objc2::AnyThread;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

impl MatMul<f32> for MetalGpu {
    fn matmul<const R: usize, const C: usize, const K: usize>(
        a: &<Self as HasStorage<f32, { R * C }>>::Storage,
        b: &<Self as HasStorage<f32, { C * K }>>::Storage,
        out: &mut <Self as HasStorage<f32, { R * K }>>::Storage,
    ) where
        Self: HasStorage<f32, { R * C }> + HasStorage<f32, { C * K }> + HasStorage<f32, { R * K }>,
    {
        unsafe {
            let dev = &MetalGpu::shared().device;
            let queue = &MetalGpu::shared().queue;

            let rb_a =
                MPSMatrixDescriptor::rowBytesForColumns_dataType(C as _, MPSDataType::Float32);
            let rb_b =
                MPSMatrixDescriptor::rowBytesForColumns_dataType(K as _, MPSDataType::Float32);
            let rb_out =
                MPSMatrixDescriptor::rowBytesForColumns_dataType(K as _, MPSDataType::Float32);

            let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                R as _,
                C as _,
                rb_a,
                MPSDataType::Float32,
            );
            let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                C as _,
                K as _,
                rb_b,
                MPSDataType::Float32,
            );
            let desc_out = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                R as _,
                K as _,
                rb_out,
                MPSDataType::Float32,
            );

            // Cast each buffer individually
            let a_buf: &MetalGpuStorage = &*(a as *const _ as *const _);
            let b_buf: &MetalGpuStorage = &*(b as *const _ as *const _);
            let out_buf: &MetalGpuStorage = &*(out as *const _ as *const _);

            let mat_a = MPSMatrix::alloc();
            let mat_a = MPSMatrix::initWithBuffer_descriptor(mat_a, &a_buf.buffer, &desc_a);

            let mat_b = MPSMatrix::alloc();
            let mat_b = MPSMatrix::initWithBuffer_descriptor(mat_b, &b_buf.buffer, &desc_b);

            let mat_out = MPSMatrix::alloc();
            let mat_out = MPSMatrix::initWithBuffer_descriptor(mat_out, &out_buf.buffer, &desc_out);

            // Use simplified or fallback constructor; use msg_send! if needed
            let mm = MPSMatrixMultiplication::alloc();
            let mm = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                mm, dev, false, false, R as _, K as _, C as _, 1.0, 0.0,
            );

            let cmd_buf = queue.commandBuffer().expect("cmd buffer");
            mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                &cmd_buf, &mat_a, &mat_b, &mat_out,
            );
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }
}

#[cfg(test)]
mod bench {
    use crate::storage::metal_gpu::MetalGpu;
    use crate::tensor::Tensor2;
    use test::Bencher;

    #[bench]
    fn matmul_128x128(b: &mut Bencher) {
        const N: usize = 256;

        let a = Tensor2::<f32, N, N, MetalGpu>::new([0.0; N * N]);
        let b_mat = Tensor2::<f32, N, N, MetalGpu>::new([0.0; N * N]);

        b.iter(|| {
            test::black_box(a.clone() * b_mat.clone());
        });
    }
}
