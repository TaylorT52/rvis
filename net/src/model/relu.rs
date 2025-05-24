use ndarray::Array4;

#[derive(Debug)]
pub struct Relu {
    input: Option<Array4<f32>>
}

impl Relu {
    pub fn new() -> Self {
        return Self {input: None}
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32>{
        let activated = input.mapv(|x| x.max(0.0));
        self.input = Some(input.clone());
        activated
    }

    pub fn backward(&self, grad_output: &Array4<f32>) -> Array4<f32> {
        let input = self.input.as_ref().expect("relu fwd must be called before relu backward");
        input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * grad_output
    } 
}