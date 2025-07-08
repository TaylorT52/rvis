use crate::storage::HasStorage;
use crate::storage::metal_gpu::{MetalGpu, MetalGpuStorage};
use crate::tensor_ops::relu::Relu;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLSize, MTLDevice,
};
use once_cell::sync::OnceCell;

impl Relu<f32> for MetalGpu {
    fn relu<const N: usize>(
        a: &<Self as HasStorage<f32, N>>::Storage,
        out: &mut <Self as HasStorage<f32, N>>::Storage,
    ) where
        Self: HasStorage<f32, N>,
    {
        unsafe {
           //compiles the kernel once, stores it in a onecell. 
            static PSO: OnceCell<*mut dyn MTLComputePipelineState> = OnceCell::new();
            let device = &MetalGpu::shared().device;
            let pso = PSO.get_or_init(|| {
                const SRC: &str = r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void relu_kernel(device float *x [[ buffer(0) ]],
                                            uint  gid      [[ thread_position_in_grid ]]) {
                        x[gid] = fmax(x[gid], 0.0f);
                    }
                "#;

                let mut err: *mut objc2::runtime::AnyObject = std::ptr::null_mut();
                let lib = device.newLibraryWithSource_options_error(SRC, std::ptr::null(), &mut err);
                if lib.is_null() {
                    panic!("Metal compilation failed: {:?}", err);
                }
                let func = lib.getFunctionWithName("relu_kernel").expect("function not found");
                device
                    .newComputePipelineStateWithFunction_error(&func, &mut err)
                    .expect("pipeline state")
                    .into_raw()
            });

            //allocates the command buffer, encoder
            let queue: &dyn MTLCommandQueue = &MetalGpu::shared().queue;
            let cmd_buf = queue.commandBuffer().expect("cmd buffer");
            let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
            encoder.setComputePipelineState(&*pso);

            //binds tensor buffer 
            let in_buf: &MetalGpuStorage = &*(a as *const _ as *const _);
            encoder.setBuffer_offset_atIndex(&in_buf.buffer, 0, 0);

            //dispatches, allocates threadgroups, and threads per threadgroup
            const TG: usize = 256;                    // threads / thread-group
            let threads_per_tg = MTLSize {            // (TG,1,1)
                width:  TG as _,
                height: 1,
                depth:  1,
            };
            let tg_count = MTLSize {                  // ceil(N / TG), 1, 1
                width:  ((N + TG - 1) / TG) as _,
                height: 1,
                depth:  1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(tg_count, threads_per_tg);

            //commites to hardware and waits
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }

    fn relu_backward<const N: usize>(
        _input: &<Self as HasStorage<f32, N>>::Storage,
        _grad_output: &<Self as HasStorage<f32, N>>::Storage,
        _grad_input: &mut <Self as HasStorage<f32, N>>::Storage,
    ) where
        Self: HasStorage<f32, N>,
    {
        // TODO: Implement GPU relu_backward
        unimplemented!("GPU relu_backward not implemented yet");
    }
}
