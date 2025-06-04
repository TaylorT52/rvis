pub struct MnistCNN {
    //conv-relu-pool x2
    c1: Conv2D, r1: Relu, p1: MaxPool2D,
    c2: Conv2D, r2: Relu, p2: MaxPool2D,

    //fc 128 -> 10
    fc1: Dense,
    r3:  Relu,
    fc2: Dense,
    softmax: Softmax,
}

impl MnistCNN {
    pub fn new() -> Self {
        //input 1×28×28  ->  32×28×28
        let c1 = Conv2D::new(1, 32, 3, 1, 1, InitScheme::HeNormal);
        //32×28×28 -> MaxPool(2) -> 32×14×14
        let p1 = MaxPool2D::new(2, 2, 0);

        //32×14×14 →-> 64×14×14
        let c2 = Conv2D::new(32, 64, 3, 1, 1, InitScheme::HeNormal);
        //maxpool -> 64×7×7
        let p2 = MaxPool2D::new(2, 2, 0);

        //flatten -> 64*7*7 = 3136
        let fc1 = Dense::new(64 * 7 * 7, 128);
        let fc2 = Dense::new(128, 10);

        Self {
            c1, r1: Relu::new(), p1,
            c2, r2: Relu::new(), p2,
            fc1, r3: Relu::new(), fc2,
            softmax: Softmax::new(1),
        }
    }

    //fwd returns probabilities [B, 10, 1, 1]
    pub fn forward(&mut self, x: &Array4<f32>) -> (Array4<f32>,  
                                                   Array2<f32>, 
                                                   Array2<f32>)
    {
        let x = self.p1.forward(&self.r1.forward(&self.c1.forward(x)));
        let x = self.p2.forward(&self.r2.forward(&self.c2.forward(&x)));

        let b = x.shape()[0];
        let flat = x.view().into_shape((b, 64 * 7 * 7)).unwrap();

        let z1 = self.r3.forward(&self.fc1.forward(&flat));
        let logits = self.fc2.forward(&z1);

        let probs = self.softmax.forward(
            &logits.insert_axis(Axis(2)).insert_axis(Axis(3))
        );
        (probs, z1, flat)
    }

    pub fn train_batch(&mut self, x: &Array4<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        let (probs, z1, flat) = self.forward(x);
        let loss = Self::cross_entropy_loss(&probs, y);

        let mut grad = self.softmax.backward(y);               
        grad = grad.index_axis_move(Axis(3), 0); 
        grad = grad.index_axis_move(Axis(2), 0);

        //fc2
        grad = self.fc2.backward(&z1, &grad); 
        grad = self.r3.backward(&grad);

        //fc1
        grad = self.fc1.backward(&flat, &grad);

        let grad = grad.into_shape((x.shape()[0], 64, 7, 7)).unwrap();

        //back through second block
        let grad = self.c2.backward(
            &self.r2.backward(&self.p2.backward(&grad))
        );

        //first block
        let _ = self.c1.backward(
            &self.r1.backward(&self.p1.backward(&grad))
        );

        //sgd update
        let bs = x.shape()[0];
        self.fc2.step(lr, bs);
        self.fc1.step(lr, bs);
        self.c2.update_weights(lr);
        self.c1.update_weights(lr);

        loss
    }

    fn cross_entropy_loss(logits: &Array4<f32>, targets: &Array2<f32>) -> f32 { /* … */ }
}
