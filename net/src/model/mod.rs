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
pub mod flatten;

pub use flatten::Flatten;
pub use self::conv::Conv2D;
pub use self::dense::Dense;
pub use self::dropout::Dropout;
pub use self::mnistclassifier::MnistCNN;
pub use self::pool::MaxPool2D;
pub use self::relu::ReluLayer;
pub use self::softmax::Softmax;
use crate::model::conv::InitScheme;
