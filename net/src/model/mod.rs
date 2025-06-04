use rand::Rng;
use ndarray::{Array1, Array2, Array4, Axis};

pub mod conv;
pub mod relu;
pub mod pool;
pub mod softmax;

use crate::model::conv::InitScheme;
pub use self::conv::Conv2D;
pub use self::relu::Relu;
pub use self::pool::MaxPool2D;
pub use self::softmax::Softmax;

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
    softmax: Softmax,
    last_flat: Option<Array2<f32>>, // Store flattened input for FC layer
}

impl SimpleCNN {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let hidden_dim = 16 * 7 * 7; // Adjusted for MNIST image size
        let output_dim = 10;

        // Use He initialization for FC layer
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let fc_weights = Array2::from_shape_fn((hidden_dim, output_dim), |_| rng.gen_range(-scale..scale));
        let fc_bias = Array1::zeros(output_dim);

        Self {
            conv1: Conv2D::new(1, 8, 3, 1, 1, InitScheme::HeNormal), // Changed input channels to 1
            relu1: Relu::new(),
            pool1: MaxPool2D::new(2, 2, 0),
            conv2: Conv2D::new(8, 16, 3, 1, 1, InitScheme::HeNormal),
            relu2: Relu::new(),
            pool2: MaxPool2D::new(2, 2, 0),
            fc_weights,
            fc_bias,
            softmax: Softmax::new(1), // final output is [B, C, 1, 1]
            last_flat: None,
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
        let flat = x.view().into_shape((batch_size, 16 * 7 * 7)).unwrap();
        self.last_flat = Some(flat.to_owned()); // Convert view to owned array before storing

        //fc 
        let logits = flat.dot(&self.fc_weights) + &self.fc_bias;

        //resh for softmax - ensure 4D shape [B, C, 1, 1]
        let reshaped = logits.insert_axis(Axis(2)).insert_axis(Axis(3));
        
        self.softmax.forward(&reshaped)
    }

    pub fn train_batch(&mut self, x: &Array4<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        let logits = self.forward(x);
        let loss = Self::cross_entropy_loss(&logits, y);
        let grad = self.softmax.backward(y);

        // Clone gradient before reshaping for FC layer
        let grad_reshaped = grad.clone().into_shape((x.shape()[0], 10)).unwrap();

        let grad = self.pool2.backward(&grad);
        let grad = self.relu2.backward(&grad);
        let grad = self.conv2.backward(&grad);

        let grad = self.pool1.backward(&grad);
        let grad = self.relu1.backward(&grad);
        let _grad = self.conv1.backward(&grad);

        // Use a smaller learning rate for the fully connected layer
        self.conv1.update_weights(lr * 0.1);
        self.conv2.update_weights(lr * 0.1);

        // Update FC layer weights and bias
        if let Some(flat) = &self.last_flat {
            // Compute gradients
            let grad_w = flat.t().dot(&grad_reshaped);
            let grad_b = grad_reshaped.sum_axis(Axis(0));
            
            // Update weights and bias
            self.fc_weights -= &(lr * &grad_w);
            self.fc_bias -= &(lr * &grad_b);
        }

        loss
    }

    pub fn cross_entropy_loss(logits: &Array4<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = logits.shape()[0];
    
        let mut loss = 0.0;
        for (logit, target) in logits.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0))) {
            // Get the channel dimension (class probabilities)
            let logit_flat = logit.into_shape((10,)).unwrap();
            
            for (p, &t) in logit_flat.iter().zip(target.iter()) {
                if t == 1.0 {
                    loss -= p.max(1e-9).ln(); //prev log0
                }
            }
        }
    
        loss / batch_size as f32
    }    
}