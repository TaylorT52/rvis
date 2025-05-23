use ndarray::{Array4, s};

#[derive(Debug)]
pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        return Self
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32>{
        input.mapv(|x| x.max(0.0))
    }
}