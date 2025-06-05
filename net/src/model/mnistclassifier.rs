use ndarray::{Array2, Array4, Axis};
use crate::model::{Conv2D, Relu, MaxPool2D, Dense, Softmax, InitScheme, BatchNorm2D, Dropout};

pub struct MnistCNN {
    //conv-relu-pool x2
    c1: Conv2D, bn1: BatchNorm2D, r1: Relu, p1: MaxPool2D,
    c2: Conv2D, bn2: BatchNorm2D, r2: Relu, p2: MaxPool2D,
    dropout1: Dropout,

    //fc 128 -> 10
    fc1: Dense,
    bn3: BatchNorm2D,
    r3: Relu,
    dropout2: Dropout,
    fc2: Dense,
    softmax: Softmax,
}

impl MnistCNN {
    pub fn new() -> Self {
        //input 1×28×28  ->  32×28×28
        let c1 = Conv2D::new(1, 32, 3, 1, 1, InitScheme::HeNormal);
        let bn1 = BatchNorm2D::new(32);
        //32×28×28 -> MaxPool(2) -> 32×14×14
        let p1 = MaxPool2D::new(2, 2, 0);

        //32×14×14 →-> 64×14×14
        let c2 = Conv2D::new(32, 64, 3, 1, 1, InitScheme::HeNormal);
        let bn2 = BatchNorm2D::new(64);
        //maxpool -> 64×7×7
        let p2 = MaxPool2D::new(2, 2, 0);

        //flatten -> 64*7*7 = 3136
        let fc1 = Dense::new(64 * 7 * 7, 256); // Increased hidden size
        let bn3 = BatchNorm2D::new(256);
        let fc2 = Dense::new(256, 10);

        Self {
            c1, bn1, r1: Relu::new(), p1,
            c2, bn2, r2: Relu::new(), p2,
            dropout1: Dropout::new(0.25), // Add dropout after conv layers
            fc1, bn3, r3: Relu::new(),
            dropout2: Dropout::new(0.5), // Add dropout after fc layer
            fc2,
            softmax: Softmax::new(1),
        }
    }

    //fwd returns probabilities [B, 10, 1, 1]
    pub fn forward(&mut self, x: &Array4<f32>) -> (Array4<f32>,  
                                                   Array2<f32>, 
                                                   Array2<f32>)
    {
        let x = self.p1.forward(&self.r1.forward(&self.bn1.forward(&self.c1.forward(x))));
        let x = self.p2.forward(&self.r2.forward(&self.bn2.forward(&self.c2.forward(&x))));
        let x = self.dropout1.forward(&x);

        let b = x.shape()[0];
        let flat = x.view().into_shape((b, 64 * 7 * 7)).unwrap().to_owned();

        let z1 = self.fc1.forward(&flat); // [B, 256]
        let z1_bn = self.bn3.forward(&z1.insert_axis(Axis(2)).insert_axis(Axis(3))); // [B, 256, 1, 1]
        let z1_relu = self.r3.forward(&z1_bn); // [B, 256, 1, 1]
        let z1_drop = self.dropout2.forward(&z1_relu); // [B, 256, 1, 1]
        let z1_flat = z1_drop.index_axis_move(Axis(3), 0).index_axis_move(Axis(2), 0); // [B, 256]
        let logits = self.fc2.forward(&z1_flat); // [B, 10]

        // Reshape logits to 4D for softmax
        let logits_4d = logits.insert_axis(Axis(2)).insert_axis(Axis(3)); // [B, 10, 1, 1]
        let probs = self.softmax.forward(&logits_4d);
        (probs, z1_flat, flat)
    }

    pub fn train_batch(&mut self, x: &Array4<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        let (probs, z1_flat, flat) = self.forward(x);
        let loss = Self::cross_entropy_loss(&probs, y);

        // grad: [B, 10, 1, 1]
        let grad_4d = self.softmax.backward(y);
        
        // Convert to 2D for fc2 by removing singleton dimensions
        let grad = grad_4d.index_axis_move(Axis(3), 0)
            .index_axis_move(Axis(2), 0)
            .to_owned();

        // fc2 backward
        let grad_fc2 = self.fc2.backward(&z1_flat, &grad); // [B, 256]
        let grad_drop = self.dropout2.backward(&grad_fc2.insert_axis(Axis(2)).insert_axis(Axis(3))); // [B, 256, 1, 1]
        let grad_relu = self.r3.backward(&grad_drop); // [B, 256, 1, 1]
        let grad_bn = self.bn3.backward(&grad_relu); // [B, 256, 1, 1]
        let grad_bn_flat = grad_bn.index_axis_move(Axis(3), 0)
            .index_axis_move(Axis(2), 0)
            .to_owned(); // [B, 256]

        // fc1 backward
        let grad_fc1 = self.fc1.backward(&flat, &grad_bn_flat); // [B, 3136]

        let grad = grad_fc1.into_shape((x.shape()[0], 64, 7, 7)).unwrap();

        //back through second block
        let grad = self.c2.backward(
            &self.r2.backward(&self.bn2.backward(&self.p2.backward(&grad)))
        );
        let grad = self.dropout1.backward(&grad);

        //first block
        let _ = self.c1.backward(
            &self.r1.backward(&self.bn1.backward(&self.p1.backward(&grad)))
        );

        //sgd update
        let bs = x.shape()[0];
        self.fc2.step(lr, bs);
        self.fc1.step(lr, bs);
        self.c2.update_weights(lr);
        self.c1.update_weights(lr);

        loss
    }

    fn cross_entropy_loss(logits: &Array4<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = logits.shape()[0];
        // Convert 4D logits to 2D using view and reshape
        let logits_2d = logits.view()
            .into_shape((batch_size, 10))
            .unwrap();
    
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
