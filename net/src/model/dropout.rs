use ndarray::{Array4, Array2};
use rand::Rng;

pub struct Dropout {
    p: f32,
    mask: Option<Array4<f32>>,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            p,
            mask: None,
        }
    }

    pub fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        let mut rng = rand::thread_rng();
        let mask = Array4::from_shape_fn(x.dim(), |_| {
            if rng.gen::<f32>() < self.p {
                0.0
            } else {
                1.0 / (1.0 - self.p)
            }
        });
        self.mask = Some(mask.clone());
        x * &mask
    }

    pub fn backward(&self, grad: &Array4<f32>) -> Array4<f32> {
        if let Some(mask) = &self.mask {
            grad * mask
        } else {
            grad.clone()
        }
    }
} 