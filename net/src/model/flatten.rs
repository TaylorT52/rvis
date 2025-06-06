use ndarray::{Array, Array2, Array4};

#[derive(Debug)]
pub struct Flatten {
    pub input_shape: Option<(usize, usize, usize, usize)>,
}

impl Flatten {
    pub fn new() -> Self {
        Self { input_shape: None }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array2<f32> {
        let (b, c, h, w) = input.dim();
        self.input_shape = Some((b, c, h, w));
        
        // Clone the input before reshaping
        input.clone().into_shape((b, c * h * w)).unwrap()
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array4<f32> {
        let (b, c, h, w) = self.input_shape.unwrap();
        
        // Clone the gradient output before reshaping
        grad_output.clone().into_shape((b, c, h, w)).unwrap()
    }
}