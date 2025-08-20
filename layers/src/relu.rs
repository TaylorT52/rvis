use tensor::tensor::Tensor3;
use tensor::storage::metal_gpu::MetalGpu;
use tensor::tensor_ops::relu::Relu;

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
        MetalGpu: Relu<f32>,
    {
        // Apply ReLU using the tensor's built-in method
        let result = input.clone().relu();
        
        // Update the input with the result
        *input = result;
    }
}