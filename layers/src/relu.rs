use tensor::tensor::Tensor3;
use tensor::storage::metal_gpu::MetalGpu;
use tensor::storage::HasStorage;

pub struct ReLU; 

impl ReLU {
    pub fn new() -> Self { 
        Self
    }

    pub fn forward<const D0: usize, const D1: usize, const D2: usize>(
        &self,
        input: &mut Tensor3<f32, D0, D1, D2, MetalGpu>,
    )
    where
        [(); D0 * (D1 * D2)]:,
        MetalGpu: HasStorage<f32, {D0 * D1 * D2}>, 
    {
        let _size = input;
    }
}