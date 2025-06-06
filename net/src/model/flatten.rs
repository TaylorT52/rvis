use tensor::tensor::{Tensor2, Tensor4};
use tensor::storage::naive_cpu::NaiveCpu;
use tensor::storage::HasStorage;

pub struct Flatten {
    input_shape: Option<(usize, usize, usize, usize)>, //(B, C, H, W)
}

impl Flatten {
    pub fn new() -> Self {
        Self {
            input_shape: None,
        }
    }

    /// Forward: Converts [B, C, H, W] -> [B, C*H*W]
    pub fn forward<const B: usize, const C: usize, const H: usize, const W: usize>(
        &mut self,
        input: &Tensor4<f32, B, C, H, W, NaiveCpu>,
    ) -> Tensor2<f32, B, { C * H * W }, NaiveCpu>
    where
        [(); B * C * H * W]:,
        [(); B * (C * H * W)]:,
    {
        self.input_shape = Some((B, C, H, W));
        
        // Get input data as slice
        let input_slice = input.as_slice();
        
        // Create output storage
        let mut output_data = <NaiveCpu as HasStorage<f32, { B * (C * H * W) }>>::storage_uninit();
        let output_slice = <NaiveCpu as HasStorage<f32, { B * (C * H * W) }>>::as_mut_slice(&mut output_data);
        
        // Since flatten just reshapes without changing memory layout,
        // we can copy directly
        output_slice.copy_from_slice(input_slice);
        
        // Create and return output tensor
        Tensor2 {
            storage: output_data,
            _p: core::marker::PhantomData,
        }
    }
    
    /// Backward: Converts gradients from [B, C*H*W] back to [B, C, H, W]
    pub fn backward<const B: usize, const C: usize, const H: usize, const W: usize>(
        &self,
        grad_output: &Tensor2<f32, B, { C * H * W }, NaiveCpu>,
    ) -> Tensor4<f32, B, C, H, W, NaiveCpu>
    where
        [(); B * C * H * W]:,
        [(); B * (C * H * W)]:,
        [(); B * (C * (H * W))]:,
    {
        // Get gradient data as slice
        let grad_slice = grad_output.as_slice();
        
        // Create output storage for gradients
        let mut grad_input_data = <NaiveCpu as HasStorage<f32, { B * C * H * W }>>::storage_uninit();
        let grad_input_slice = <NaiveCpu as HasStorage<f32, { B * C * H * W }>>::as_mut_slice(&mut grad_input_data);
        
        // Copy gradients (reshape doesn't change values, just interpretation)
        grad_input_slice.copy_from_slice(grad_slice);
        
        // Create and return gradient tensor
        Tensor4 {
            storage: grad_input_data,
            _p: core::marker::PhantomData,
        }
    }
}