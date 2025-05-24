pub mod conv;
pub mod relu;
pub mod pool;
pub mod softmax;

use crate::model::conv::InitScheme;
pub use self::conv::Conv2D;
pub use self::relu::Relu;
pub use self::pool::MaxPool2D;
pub use self::softmax::Softmax;

use ndarray::Array4;

#[derive(Debug)]
pub struct SimpleCNN {
    conv1: Conv2D, 
    relu1: Relu, 
    pool1: MaxPool2D, 
    conv2: Conv2D, 
    relu2: Relu, 
    pool2: MaxPool2D,
    softmax: Softmax
}

impl SimpleCNN {
    pub fn new() -> Self {
        Self {
            conv1: Conv2D::new(3, 8, 3, 1, 1, InitScheme::HeNormal),
            relu1: Relu::new(),
            pool1: MaxPool2D::new(2, 2, 0),
            conv2: Conv2D::new(8, 16, 3, 1, 1, InitScheme::HeNormal),
            relu2: Relu::new(),
            pool2: MaxPool2D::new(2, 2, 0),
            softmax: Softmax::new(1), // final output is [B, C, 1, 1]
        }
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let x = self.conv1.forward(input);
        let x = self.relu1.forward(&x);
        let x = self.pool1.forward(&x);
        let x = self.conv2.forward(&x);
        let x = self.relu2.forward(&x);
        let x = self.pool2.forward(&x);
        self.softmax.forward(&x)
    }
}