// src/tensor_ops/relu.rs
use crate::storage::{HasStorage, metal_gpu::{MetalGpu, MetalGpuStorage}};

use objc2::rc::Id;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLSize,
    // choose the exact imports your objc2_metal version exposes
};
use once_cell::sync::OnceCell;

/// A tiny trait that every backend implements if it *can* do ReLU in-place.
pub trait ReluKernel<T> {
    /// Apply ReLU to `buf` in-place. Length is encoded in the const generic.
    fn relu<const N: usize>(buf: &mut <Self as HasStorage<T, N>>::Storage)
    where
        Self: HasStorage<T, N>;
}

/* ---------- Metal implementation ---------- */

impl ReluKernel<f32> for MetalGpu {
    fn relu<const N: usize>(buf: &mut <Self as HasStorage<f32, N>>::Storage)
    where
        Self: HasStorage<f32, N>,
    {
        unsafe {
            /* --------- one-time pipeline compilation --------- */
            static PSO: OnceCell<Id<MTLComputePipelineState>> = OnceCell::new();
            let device = &MetalGpu::shared().device;
            let pso = PSO.get_or_init(|| {
                // You can also pre-compile the .metallib and bundle it.
                let src = include_str!("relu.metal");
                let opts = std::ptr::null(); // default compile options
                let mut err: *mut objc2::runtime::Object = std::ptr::null_mut();
                // newLibraryWithSource:options:error:
                let lib = device.newLibraryWithSource_options_error(src, opts, &mut err);
                if lib.is_null() {
                    panic!("Metal compile error: {:?}", err);
                }
                let func = lib.getFunctionWithName("relu_kernel").expect("fn");
                device
                    .newComputePipelineStateWithFunction_error(&func, &mut err)
                    .expect("pipeline")
            });

            /* --------- command buffer & encoder --------- */
            let queue: &Id<MTLCommandQueue> = &MetalGpu::shared().queue;
            let cmd_buf = queue.commandBuffer().expect("cmd buffer");
            let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
            encoder.setComputePipelineState(pso);

            /* --------- bind the tensor buffer --------- */
            let storage: &MetalGpuStorage = &*(buf as *mut _ as *mut _);
            encoder.setBuffer_offset_atIndex(&storage.buffer, 0, 0);

            /* --------- dispatch --------- */
            const TG_SIZE: usize = 256;
            let thr_per_tg = MTLSize {
                width: TG_SIZE as _,
                height: 1,
                depth: 1,
            };
            let tg_count = MTLSize {
                width: ((N + TG_SIZE - 1) / TG_SIZE) as _,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(tg_count, thr_per_tg);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }
}
