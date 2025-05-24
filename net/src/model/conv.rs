use ndarray::{Array, Array1, Array4, s};
use rand::Rng;
use ndarray_rand::rand_distr::{Distribution, Normal};

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
    pub input: Option<Array4<f32>>
}

//give user more finegrained control over weight init
pub enum InitScheme {
    Xavier,
    HeNormal,
    Constant(f32),
    RandomUniform(f32),
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
        let mut rng = rand::thread_rng();

        let (fan_in, fan_out) =
            (in_channels * kernel_size * kernel_size,
             out_channels * kernel_size * kernel_size);

        //choose weight tensor based on init scheme
        let weights = match scheme {
            InitScheme::Xavier => {
                let lim = (6.0 / (fan_in + fan_out) as f32).sqrt();
                Array::from_shape_fn(
                    (out_channels, in_channels, kernel_size, kernel_size),
                    |_| rng.gen_range(-lim..lim),
                )
            }

            //diff weight inits based on input schemes
            InitScheme::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                Array::from_shape_fn(
                    (out_channels, in_channels, kernel_size, kernel_size),
                    |_| normal.sample(&mut rng),
                )
            }

            InitScheme::Constant(c) => {
                Array::from_elem(
                    (out_channels, in_channels, kernel_size, kernel_size),
                    c,
                )
            }

            InitScheme::RandomUniform(scale) => {
                Array::from_shape_fn(
                    (out_channels, in_channels, kernel_size, kernel_size),
                    |_| rng.gen_range(-scale..scale),
                )
            }
        };

        //init bias as 1d arr 0's
        let bias = Array::zeros(out_channels);

        let grad_weights = Array::zeros((out_channels, in_channels, kernel_size, kernel_size));
        let grad_bias = Array::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            grad_weights,
            grad_bias,
            input: None,
        }
    }

    pub fn pad_input(&self, input: &Array4<f32>) -> Array4<f32> {
        if self.padding == 0 {
            input.clone()
        } else {
            let (b, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
            let new_h = h + 2 * self.padding; 
            let new_w = w + 2 * self.padding; 
            let mut padded = Array4::<f32>::zeros((b, c, new_h, new_w)); 
            padded.slice_mut(s![.., ..,
                self.padding..self.padding + h,
                self.padding..self.padding + w])
            .assign(input);
            padded
        }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        self.input = Some(input.clone());
        let input = self.pad_input(&input); 
        //check channel count
        assert_eq!(
            input.shape()[1],
            self.in_channels,
            "Conv2D: input has {} channels, but layer expects {}",
            input.shape()[1],
            self.in_channels
        );

        let (batch, _c, h_in_p, w_in_p) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );

        //output spatial dimensions
        let out_h = (h_in_p - self.kernel_size) / self.stride + 1;
        let out_w = (w_in_p - self.kernel_size) / self.stride + 1;

        //alloc outptut 
        let mut output =
            Array4::<f32>::zeros((batch, self.out_channels, out_h, out_w));

        //batch → out_ch → out_row → out_col
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = self.bias[oc];

                        //inner prod vs patch
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    sum += input[[b, ic, ih, iw]]
                                        * self.weights[[oc, ic, kh, kw]];
                                }
                            }
                        }

                        output[[b, oc, oh, ow]] = sum;
                    }
                }
            }
        }

        output
    }

    //backward pass
    pub fn backward(&mut self, grad_output: &Array4<f32>) -> Array4<f32> {
        let input = self.input.as_ref().expect("Conv2D: forward() must be called before backward()");
        let input_padded = self.pad_input(&input);
        let (batch_size, _, h_in, w_in) = input_padded.dim();
        let (_, _, h_out, w_out) = grad_output.dim();

        let mut grad_input_padded = Array4::<f32>::zeros(input_padded.dim());
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let val = grad_output[[b, oc, i, j]];
                        self.grad_bias[oc] += val;
    
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let h = i * self.stride + kh;
                                    let w = j * self.stride + kw;
    
                                    let input_val = input_padded[[b, ic, h, w]];
                                    self.grad_weights[[oc, ic, kh, kw]] += input_val * val;
                                    grad_input_padded[[b, ic, h, w]] += self.weights[[oc, ic, kh, kw]] * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    
        if self.padding > 0 {
            let start = self.padding;
            let end_h = start + input.shape()[2];
            let end_w = start + input.shape()[3];
            grad_input_padded.slice_move(s![.., .., start..end_h, start..end_w])
        } else {
            grad_input_padded
        }
    }    
}
