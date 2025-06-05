use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array1, Array2, Array4, Axis};
use rand::Rng;
use std::fs::File;
use std::io;

pub mod conv;
pub mod dense;
pub mod dropout;
pub mod mnistclassifier;
pub mod pool;
pub mod relu;
pub mod softmax;

pub use self::conv::{BatchNorm2D, Conv2D, LeakyReLU, Route, Shortcut, Upsample};
pub use self::dense::Dense;
pub use self::dropout::Dropout;
pub use self::mnistclassifier::MnistCNN;
pub use self::pool::MaxPool2D;
pub use self::relu::Relu;
pub use self::softmax::Softmax;
use crate::model::conv::InitScheme;

#[derive(Debug)]
pub struct SimpleCNN {
    conv1: Conv2D,
    relu1: Relu,
    pool1: MaxPool2D,
    conv2: Conv2D,
    relu2: Relu,
    pool2: MaxPool2D,
    fc_weights: Array2<f32>,
    fc_bias: Array1<f32>,
    last_flat_input: Option<Array2<f32>>,
    softmax: Softmax,
}

impl SimpleCNN {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let hidden_dim = 16 * 7 * 7; // Changed from 16 * 8 * 8 to match actual output shape
        let output_dim = 10;

        Self {
            conv1: Conv2D::new(1, 8, 3, 1, 1, InitScheme::HeNormal),
            relu1: Relu::new(),
            pool1: MaxPool2D::new(2, 2, 0),
            conv2: Conv2D::new(8, 16, 3, 1, 1, InitScheme::HeNormal),
            relu2: Relu::new(),
            pool2: MaxPool2D::new(2, 2, 0),
            fc_weights: Array2::from_shape_fn((hidden_dim, output_dim), |_| {
                rng.gen_range(-0.1..0.1)
            }),
            fc_bias: Array1::zeros(output_dim),
            last_flat_input: None,
            softmax: Softmax::new(1), //final output is [B, C, 1, 1]
        }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        let x = self.conv1.forward(input);
        let x = self.relu1.forward(&x);
        let x = self.pool1.forward(&x);
        let x = self.conv2.forward(&x);
        let x = self.relu2.forward(&x);
        let x = self.pool2.forward(&x);

        //flatten
        let batch_size = x.shape()[0];
        let flat = x.view().into_shape((batch_size, 16 * 8 * 8)).unwrap();

        //fc
        let logits = flat.dot(&self.fc_weights) + &self.fc_bias;

        //resh for softmax
        let reshaped = logits.insert_axis(Axis(2)).insert_axis(Axis(3));

        self.last_flat_input = Some(flat.to_owned());
        self.softmax.forward(&reshaped)
    }

    pub fn train_batch(&mut self, x: &Array4<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        //forward pass
        let batch_size = x.shape()[0];
        let logits4d = self.forward(x);
        let logits2d = logits4d.view().into_shape((batch_size, 10)).unwrap();
        let loss = Self::cross_entropy_loss(&logits4d, y);

        let grad2d = &logits2d - y;

        //fc grads
        let flat_input = self
            .last_flat_input
            .as_ref()
            .expect("forward must run first");

        let grad_w = flat_input.t().dot(&grad2d);
        let grad_b = grad2d.sum_axis(Axis(0));

        let mut grad_h = grad2d.dot(&self.fc_weights.t());
        let mut grad_4d = grad_h.into_shape((batch_size, 16, 8, 8)).unwrap();

        grad_4d = self.pool2.backward(&grad_4d);
        grad_4d = self.relu2.backward(&grad_4d);
        grad_4d = self.conv2.backward(&grad_4d);
        grad_4d = self.pool1.backward(&grad_4d);
        grad_4d = self.relu1.backward(&grad_4d);
        let _ = self.conv1.backward(&grad_4d);

        //SGD updates
        self.fc_weights -= &(lr * grad_w);
        self.fc_bias -= &(lr * grad_b);
        self.conv1.update_weights(lr);
        self.conv2.update_weights(lr);

        loss
    }

    pub fn cross_entropy_loss(logits: &Array4<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = logits.shape()[0];

        let mut loss = 0.0;
        for (logit, target) in logits.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0))) {
            for (p, &t) in logit.iter().zip(target.iter()) {
                if t == 1.0 {
                    loss -= (p + 1e-9).ln();
                }
            }
        }

        loss / batch_size as f32
    }
}

impl BatchNorm2D {
    pub fn load_weights(&mut self, file: &mut File) -> io::Result<()> {
        // Load gamma (scale)
        for i in 0..self.num_features {
            self.gamma[i] = file.read_f32::<LittleEndian>()?;
        }
        // Load beta (shift)
        for i in 0..self.num_features {
            self.beta[i] = file.read_f32::<LittleEndian>()?;
        }
        // Load mean
        for i in 0..self.num_features {
            self.mean[i] = file.read_f32::<LittleEndian>()?;
        }
        // Load variance
        for i in 0..self.num_features {
            self.var[i] = file.read_f32::<LittleEndian>()?;
        }
        Ok(())
    }
}
