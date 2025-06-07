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
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
    {
        self.input_shape = Some((B, C, H, W));
        
        // Create output storage
        let mut output_data = <NaiveCpu as HasStorage<f32, { B * (C * H * W) }>>::storage_uninit();
        let output_slice = NaiveCpu::as_mut_slice(&mut output_data);
        
        // For each batch, slice the 3D tensor (C, H, W) and flatten it
        for b in 0..B {
            // Get the 3D slice for this batch
            let batch_slice = input.slice::<0, b, b + 1>();
            
            // Flatten the 3D slice into a 1D array
            let flat_slice = batch_slice.slice::<0, 0, C>().slice::<1, 0, H>().slice::<2, 0, W>();
            
            // Copy the flattened data to the output
            let src = NaiveCpu::as_slice(&flat_slice.storage);
            let dst_start = b * (C * H * W);
            output_slice[dst_start..dst_start + (C * H * W)].copy_from_slice(src);
        }
        
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
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
        [(); B * C * H * W]:,
    {
        // Create output storage for gradients
        let mut grad_input_data = <NaiveCpu as HasStorage<f32, { B * C * H * W }>>::storage_uninit();
        let grad_input_slice = NaiveCpu::as_mut_slice(&mut grad_input_data);
        
        // For each batch, reshape the gradient slice back to 3D
        for b in 0..B {
            // Get the gradient slice for this batch
            let grad_slice = grad_output.slice::<0, b, b + 1>();
            
            // Reshape the gradient slice back to 3D
            let src = NaiveCpu::as_slice(&grad_slice.storage);
            let dst_start = b * (C * H * W);
            
            // Copy the gradient data back to the input shape
            for c in 0..C {
                for h in 0..H {
                    for w in 0..W {
                        let src_idx = c * H * W + h * W + w;
                        let dst_idx = dst_start + c * H * W + h * W + w;
                        grad_input_slice[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        
        // Create and return gradient tensor
        Tensor4 {
            storage: grad_input_data,
            _p: core::marker::PhantomData,
        }
    }
}