use crate::storage::HasStorage;
use objc2::__framework_prelude::ProtocolObject;
use objc2::rc::Retained;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use std::sync::LazyLock;
use std::{ffi::c_void, ptr::NonNull};

pub struct MetalGpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// ─── Metal guarantees these two handles are thread‑safe ──────────────────────
unsafe impl Send for MetalGpu {}
unsafe impl Sync for MetalGpu {}

static METAL_GPU_CONTEXT: LazyLock<MetalGpu> = LazyLock::new(|| {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device!");
    let queue = device.newCommandQueue().expect("No Metal queue!");

    MetalGpu { device, queue }
});

impl MetalGpu {
    #[inline]
    pub fn shared() -> &'static Self {
        &METAL_GPU_CONTEXT
    }
}

#[derive(Clone)]
pub struct MetalGpuStorage {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
}

impl<const N: usize> HasStorage<f32, N> for MetalGpu {
    type Storage = MetalGpuStorage;

    fn storage_from_array(src: [f32; N]) -> Self::Storage {
        let len_bytes = N * core::mem::size_of::<f32>();
        let device = &MetalGpu::shared().device;

        let ptr = NonNull::<c_void>::new(src.as_ptr() as *mut c_void).unwrap();
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    len_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer alloc")
        };

        MetalGpuStorage { buffer, len: N }
    }

    fn storage_uninit() -> Self::Storage {
        let len_bytes = N * core::mem::size_of::<f32>();
        let device = &MetalGpu::shared().device;

        let buffer = device
            .newBufferWithLength_options(len_bytes, MTLResourceOptions::StorageModeShared)
            .expect("buffer alloc");

        MetalGpuStorage { buffer, len: N }
    }

    fn storage_zeroes() -> Self::Storage {
        let len_bytes = N * core::mem::size_of::<f32>();
        let device = &MetalGpu::shared().device;

        let vec = vec![0.0_f32; N];
        let ptr = NonNull::<c_void>::new(vec.as_ptr() as *mut c_void).unwrap();
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    len_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer alloc")
        };

        MetalGpuStorage { buffer, len: N }
    }

    fn storage_ones() -> Self::Storage {
        let len_bytes = N * core::mem::size_of::<f32>();
        let device = &MetalGpu::shared().device;

        let vec = vec![1.0_f32; N];
        let ptr = NonNull::<c_void>::new(vec.as_ptr() as *mut c_void).unwrap();
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    len_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer alloc")
        };

        MetalGpuStorage { buffer, len: N }
    }

    fn storage_full(val: f32) -> Self::Storage {
        let len_bytes = N * core::mem::size_of::<f32>();
        let device = &MetalGpu::shared().device;

        let vec = vec![val; N];
        let ptr = NonNull::<c_void>::new(vec.as_ptr() as *mut c_void).unwrap();
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    len_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer alloc")
        };

        MetalGpuStorage { buffer, len: N }
    }

    fn as_slice(_: &Self::Storage) -> &[f32] {
        panic!("CPU access to GPU buffer")
    }
    fn as_mut_slice(_: &mut Self::Storage) -> &mut [f32] {
        panic!("mut CPU access")
    }
}
