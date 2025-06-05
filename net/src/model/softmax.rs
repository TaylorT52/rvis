use ndarray::{s, Array2, Array4};

#[derive(Debug)]

pub struct Softmax {
    pub axis: usize,
    output: Option<Array4<f32>>,
}

//supports axis 1 2 3 (channels, height, width)
impl Softmax {
    pub fn new(axis: usize) -> Self {
        Self { axis, output: None }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        let mut output = input.clone();

        //apply softmax along the specfied axis
        output.outer_iter_mut().for_each(|sample| {
            let mut view = sample.into_dimensionality::<ndarray::Ix3>().unwrap();

            match self.axis {
                0 => panic!("Softmax along batch axis not supported"),
                1 => {
                    for h in 0..view.shape()[1] {
                        for w in 0..view.shape()[2] {
                            let mut col = view.slice_mut(s![.., h, w]);
                            Self::apply_softmax(&mut col);
                        }
                    }
                }
                2 => {
                    for c in 0..view.shape()[0] {
                        for w in 0..view.shape()[2] {
                            let mut row = view.slice_mut(s![c, .., w]);
                            Self::apply_softmax(&mut row);
                        }
                    }
                }
                3 => {
                    for c in 0..view.shape()[0] {
                        for h in 0..view.shape()[1] {
                            let mut row = view.slice_mut(s![c, h, ..]);
                            Self::apply_softmax(&mut row);
                        }
                    }
                }
                _ => panic!("invalid softmax axis"),
            }
        });

        self.output = Some(output.clone());
        output
    }

    //helper to apply softmax along given arr slice
    fn apply_softmax(slice: &mut ndarray::ArrayViewMut1<f32>) {
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for val in slice.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        for val in slice.iter_mut() {
            *val /= sum;
        }
    }

    pub fn backward(&self, target: &Array2<f32>) -> Array4<f32> {
        let output = self
            .output
            .as_ref()
            .expect("softmax fwd has to be previously implemented");

        println!("output shape: {:?}", self.output.as_ref().unwrap().shape());
        println!("target shape: {:?}", target.shape());
        output - target
    }
}
