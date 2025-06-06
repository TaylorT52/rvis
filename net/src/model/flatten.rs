use ndarray::{Array2, Array4, Axis};

#[derive(Debug)]
pub struct Flatten {
    input_shape: Option<(usize, usize, usize, usize)>, // (B, C, H, W)
}

impl Flatten {
    pub fn new() -> Self {
        Self {
            input_shape: None,
        }
    }

    /// Forward: Converts [B, C, H, W] -> [B, C*H*W]
    pub fn forward(&mut self, input: &Array4<f32>) -> Array2<f32> {
        let shape = input.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        self.input_shape = Some((b, c, h, w));

        input.view()
            .into_shape((b, c * h * w))
            .expect("Failed to flatten input")
            .to_owned()
    }

    /// Backward: Converts [B, C*H*W] -> [B, C, H, W]
    pub fn backward(&self, grad_output: &Array2<f32>) -> Array4<f32> {
        let (b, c, h, w) = self
            .input_shape
            .expect("Flatten: forward must be called before backward");

        grad_output
            .view()
            .into_shape((b, c, h, w))
            .expect("Failed to reshape gradient in Flatten")
            .to_owned()
    }
}
