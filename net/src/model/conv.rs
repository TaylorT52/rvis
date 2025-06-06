use ndarray::{s, Array, Array1, Array4};
use ndarray_rand::rand_distr::{Distribution, Normal};
use rand::Rng;

#[derive(Debug)]
pub struct Conv2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weights: Array4<f32>,
    pub bias: Array1<f32>,
    pub grad_weights: Array4<f32>,
    pub grad_bias: Array1<f32>,
    pub input: Option<Array4<f32>>,
}

pub enum InitScheme {
    Xavier,
    HeNormal,
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        scheme: InitScheme,
    ) -> Self {
        let (fan_in, _) = (in_channels * kernel_size * kernel_size, out_channels);
        let mut rng = rand::thread_rng();

        let weights = match scheme {
            InitScheme::Xavier => {
                let limit = (6.0 / fan_in as f32).sqrt();
                Array::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| {
                    rng.gen_range(-limit..limit)
                })
            }
            InitScheme::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                Array::from_shape_fn((out_channels, in_channels, kernel_size, kernel_size), |_| {
                    normal.sample(&mut rng)
                })
            }
        };

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias: Array::zeros(out_channels),
            grad_weights: Array::zeros((out_channels, in_channels, kernel_size, kernel_size)),
            grad_bias: Array::zeros(out_channels),
            input: None,
        }
    }

    fn pad_input(&self, input: &Array4<f32>) -> Array4<f32> {
        if self.padding == 0 {
            input.clone()
        } else {
            let (b, c, h, w) = input.dim();
            let mut padded = Array4::zeros((b, c, h + 2 * self.padding, w + 2 * self.padding));
            padded
                .slice_mut(s![.., .., self.padding..self.padding + h, self.padding..self.padding + w])
                .assign(input);
            padded
        }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        self.input = Some(input.clone());
        let input = self.pad_input(input);
        let (b, _, h, w) = input.dim();
        let out_h = (h - self.kernel_size) / self.stride + 1;
        let out_w = (w - self.kernel_size) / self.stride + 1;

        let mut output = Array4::<f32>::zeros((b, self.out_channels, out_h, out_w));

        for n in 0..b {
            for oc in 0..self.out_channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let mut sum = self.bias[oc];
                        for ic in 0..self.in_channels {
                            for ki in 0..self.kernel_size {
                                for kj in 0..self.kernel_size {
                                    let hi = i * self.stride + ki;
                                    let wj = j * self.stride + kj;
                                    sum += input[[n, ic, hi, wj]] * self.weights[[oc, ic, ki, kj]];
                                }
                            }
                        }
                        output[[n, oc, i, j]] = sum;
                    }
                }
            }
        }

        output
    }

    pub fn backward(&mut self, grad_output: &Array4<f32>) -> Array4<f32> {
        let input = self.input.as_ref().unwrap();
        let input_padded = self.pad_input(input);
        let (b, _, h, w) = input_padded.dim();
        let (_, _, out_h, out_w) = grad_output.dim();

        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);

        let mut grad_input_padded = Array4::<f32>::zeros(input_padded.raw_dim());

        for n in 0..b {
            for oc in 0..self.out_channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let grad_val = grad_output[[n, oc, i, j]];
                        self.grad_bias[oc] += grad_val;

                        for ic in 0..self.in_channels {
                            for ki in 0..self.kernel_size {
                                for kj in 0..self.kernel_size {
                                    let hi = i * self.stride + ki;
                                    let wj = j * self.stride + kj;

                                    self.grad_weights[[oc, ic, ki, kj]] +=
                                        input_padded[[n, ic, hi, wj]] * grad_val;
                                    grad_input_padded[[n, ic, hi, wj]] +=
                                        self.weights[[oc, ic, ki, kj]] * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.padding == 0 {
            grad_input_padded
        } else {
            grad_input_padded.slice_move(s![
                ..,
                ..,
                self.padding..self.padding + input.shape()[2],
                self.padding..self.padding + input.shape()[3]
            ])
        }
    }

    pub fn update_weights(&mut self, lr: f32) {
        self.weights -= &(lr * &self.grad_weights);
        self.bias -= &(lr * &self.grad_bias);
    }
}
