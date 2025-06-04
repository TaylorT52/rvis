use ndarray::{Array1, Array2, Axis};
use rand::Rng;

//fc layer Wx + b with He‑normal init
#[derive(Debug)]
pub struct Dense {
    w: Array2<f32>, //[in, out]
    b: Array1<f32>,   // [out]
    grad_w: Array2<f32>,
    grad_b: Array1<f32>,
}

impl Dense {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_dim as f32).sqrt();
        let w = Array2::from_shape_fn((in_dim, out_dim), |_| rng.gen_range(-scale..scale));
        Self {
            w,
            b: Array1::zeros(out_dim),
            grad_w: Array2::zeros((in_dim, out_dim)),
            grad_b: Array1::zeros(out_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.w) + &self.b
    }

    pub fn backward(&mut self, x: &Array2<f32>, grad_out: &Array2<f32>) -> Array2<f32> {
        self.grad_w += &x.t().dot(grad_out);
        self.grad_b += &grad_out.sum_axis(Axis(0));

        grad_out.dot(&self.w.t())
    }

    pub fn step(&mut self, lr: f32, batch_size: usize) {
        let scale = lr / batch_size as f32;
        self.w -= &(self.grad_w.mapv(|g| g * scale));
        self.b -= &(self.grad_b.mapv(|g| g * scale));
        self.grad_w.fill(0.0);
        self.grad_b.fill(0.0);
    }
}
