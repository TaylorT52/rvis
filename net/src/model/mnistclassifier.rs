use crate::model::{Conv2D, Dense, Flatten, ReluLayer, Softmax, InitScheme};
use ndarray::{Array2, Array4, Axis};

pub struct MnistCNN {
    c1: Conv2D,
    r1: ReluLayer,
    c2: Conv2D,
    r2: ReluLayer,
    flatten: Flatten,
    fc1: Dense,
    softmax: Softmax,
}

impl MnistCNN {
    pub fn new() -> Self {
        let c1 = Conv2D::new(1, 64, 3, 1, 0, InitScheme::HeNormal);
        let c2 = Conv2D::new(64, 32, 3, 1, 0, InitScheme::HeNormal);
        let fc1 = Dense::new(32 * 24 * 24, 10);

        Self {
            c1,
            r1: ReluLayer::<1, 64, 28, 28>::new(),
            c2,
            r2: ReluLayer::<1, 32, 28, 28>::new(),
            flatten: Flatten::new(),
            fc1,
            softmax: Softmax::new(1),
        }
    }

    pub fn forward(&mut self, x: &Array4<f32>) -> (Array4<f32>, Array2<f32>) {
        let x = self.r1.forward(&self.c1.forward(x));
        let x = self.r2.forward(&self.c2.forward(&x));
        let flat = self.flatten.forward(&x);
        let logits = self.fc1.forward(&flat);
        let probs = self.softmax.forward(&logits.insert_axis(Axis(2)).insert_axis(Axis(3)));
        (probs, flat)
    }

    pub fn train_batch(&mut self, x: &Array4<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        let (probs, flat) = self.forward(x);
        let loss = Self::cross_entropy_loss(&probs, y);

        let grad_softmax = self.softmax.backward(y);
        let grad_logits = grad_softmax
            .index_axis_move(Axis(3), 0)
            .index_axis_move(Axis(2), 0)
            .to_owned();

        let grad_fc1 = self.fc1.backward(&flat, &grad_logits);
        let grad_unflattened = self.flatten.backward(&grad_fc1);

        let grad_c2 = self.c2.backward(&self.r2.backward(&grad_unflattened));
        let _ = self.c1.backward(&self.r1.backward(&grad_c2));

        let bs = x.shape()[0];
        self.fc1.step(lr, bs);
        self.c2.update_weights(lr);
        self.c1.update_weights(lr);

        loss
    }

    fn cross_entropy_loss(logits: &Array4<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = logits.shape()[0];
        let logits_2d = logits.view().into_shape((batch_size, 10)).unwrap();

        let mut loss = 0.0;
        for (logit, target) in logits_2d.axis_iter(Axis(0)).zip(targets.axis_iter(Axis(0))) {
            for (p, &t) in logit.iter().zip(target.iter()) {
                if t == 1.0 {
                    loss -= (p + 1e-9).ln();
                }
            }
        }

        loss / batch_size as f32
    }
}
