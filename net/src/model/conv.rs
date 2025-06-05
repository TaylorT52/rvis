use ndarray::{Array, Array1, Array4, s};
use rand::Rng;
use ndarray_rand::rand_distr::{Distribution, Normal};
use std::io;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use ndarray::{Axis, stack, concatenate};
use rayon::prelude::*;
use std::sync::Mutex;

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
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub mean: Array1<f32>,
    pub variance: Array1<f32>,
    pub use_batch_norm: bool,
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
            gamma: Array::zeros(out_channels),
            beta: Array::zeros(out_channels),
            mean: Array::zeros(out_channels),
            variance: Array::zeros(out_channels),
            use_batch_norm: true,  // Default to using batch norm
        }
    }

    pub fn update_weights(&mut self, lr: f32) {
        self.weights -= &(lr * &self.grad_weights);
        self.bias -= &(lr * &self.grad_bias);
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

        //alloc output 
        let mut output = Mutex::new(Array4::<f32>::zeros((batch, self.out_channels, out_h, out_w)));

        //batch → out_ch → out_row → out_col
        (0..batch).into_par_iter().for_each(|b| {
            (0..self.out_channels).into_par_iter().for_each(|c| {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let mut sum = self.bias[c];

                        //inner prod vs patch
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let ih = i * self.stride + kh;
                                    let iw = j * self.stride + kw;

                                    // Validate indices
                                    assert!(ih < h_in_p, "ih {} >= h_in_p {}", ih, h_in_p);
                                    assert!(iw < w_in_p, "iw {} >= w_in_p {}", iw, w_in_p);

                                    sum += input[[b, ic, ih, iw]]
                                        * self.weights[[c, ic, kh, kw]];
                                }
                            }
                        }

                        let mut output = output.lock().unwrap();
                        output[[b, c, i, j]] = sum;
                    }
                }
            });
        });

        let mut output = output.into_inner().unwrap();
        
        // Apply activation
        if self.use_batch_norm {
            output.mapv_inplace(|x| x.max(0.0));  // ReLU
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

    pub fn load_weights(&mut self, file: &mut File) -> io::Result<()> {
        // Load weights
        let weight_size = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size;
        let mut weights = vec![0.0f32; weight_size];
        for i in 0..weight_size {
            weights[i] = file.read_f32::<LittleEndian>()?;
        }
        
        // Reshape weights to 4D array (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = Array4::from_shape_vec(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
            weights
        ).unwrap();
        
        // Load biases
        for i in 0..self.out_channels {
            self.bias[i] = file.read_f32::<LittleEndian>()?;
        }
        
        Ok(())
    }
}

#[derive(Debug)]
pub struct BatchNorm2D {
    pub num_features: usize,
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub mean: Array1<f32>,
    pub var: Array1<f32>,
    pub running_mean: Array1<f32>,
    pub running_var: Array1<f32>,
    pub momentum: f32,
    pub eps: f32,
}

impl BatchNorm2D {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            mean: Array1::zeros(num_features),
            var: Array1::ones(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum: 0.1,
            eps: 1e-5,
        }
    }

    pub fn forward(&mut self, input: &Array4<f32>) -> Array4<f32> {
        // (B, C, H, W)
        let mut output = input.clone();
        let (batch_size, channels, height, width) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        
        // Calculate mean and variance for each channel
        for c in 0..channels {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let n = batch_size * height * width;
            
            // Calculate mean
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum += input[[b, c, h, w]];
                    }
                }
            }
            let mean = sum / n as f32;
            
            // Calculate variance
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let diff = input[[b, c, h, w]] - mean;
                        sum_sq += diff * diff;
                    }
                }
            }
            let var = sum_sq / n as f32;
            
            // Update running statistics
            self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] + self.momentum * mean;
            self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] + self.momentum * var;
            
            // Store current batch statistics
            self.mean[c] = mean;
            self.var[c] = var;
            
            // Normalize and scale
            let std = (var + self.eps).sqrt();
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        output[[b, c, h, w]] = self.gamma[c] * (input[[b, c, h, w]] - mean) / std + self.beta[c];
                    }
                }
            }
        }
        output
    }

    pub fn backward(&mut self, grad_output: &Array4<f32>) -> Array4<f32> {
        let (batch_size, channels, height, width) = (
            grad_output.shape()[0],
            grad_output.shape()[1],
            grad_output.shape()[2],
            grad_output.shape()[3],
        );
        let mut grad_input = Array4::<f32>::zeros(grad_output.dim());
        let n = batch_size * height * width;

        for c in 0..channels {
            let mean = self.mean[c];
            let var = self.var[c];
            let std = (var + self.eps).sqrt();
            let gamma = self.gamma[c];

            // Compute gradients
            let mut grad_gamma = 0.0;
            let mut grad_beta = 0.0;
            let mut grad_x_hat = Array4::<f32>::zeros((batch_size, 1, height, width));

            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let x_hat = (grad_output[[b, c, h, w]] - mean) / std;
                        grad_x_hat[[b, 0, h, w]] = x_hat;
                        grad_gamma += x_hat * grad_output[[b, c, h, w]];
                        grad_beta += grad_output[[b, c, h, w]];
                    }
                }
            }

            // Update parameters
            self.gamma[c] -= 0.001 * grad_gamma;
            self.beta[c] -= 0.001 * grad_beta;

            // Compute gradient with respect to input
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let x_hat = (grad_output[[b, c, h, w]] - mean) / std;
                        grad_input[[b, c, h, w]] = gamma * (grad_output[[b, c, h, w]] - grad_gamma / n as f32 - x_hat * grad_x_hat[[b, 0, h, w]] / n as f32) / std;
                    }
                }
            }
        }
        grad_input
    }
}

#[derive(Debug)]
pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        input.mapv(|x| if x > 0.0 { x } else { self.negative_slope * x })
    }
}

#[derive(Debug)]
pub struct Upsample {
    pub stride: usize,
}

impl Upsample {
    pub fn new(stride: usize) -> Self {
        Self { stride }
    }
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (b, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let new_h = h * self.stride;
        let new_w = w * self.stride;
        let mut output = Array4::<f32>::zeros((b, c, new_h, new_w));
        for batch in 0..b {
            for ch in 0..c {
                for i in 0..h {
                    for j in 0..w {
                        for di in 0..self.stride {
                            for dj in 0..self.stride {
                                output[[batch, ch, i * self.stride + di, j * self.stride + dj]] = input[[batch, ch, i, j]];
                            }
                        }
                    }
                }
            }
        }
        output
    }
}

#[derive(Debug)]
pub struct Route {
    pub layers: Vec<isize>,
    upsample: Option<Upsample>,  // Optional upsampling layer
    upsample_idx: Option<usize>, // Which input to upsample
}

impl Route {
    pub fn new(layers: Vec<isize>) -> Self {
        Self { 
            layers,
            upsample: None,
            upsample_idx: None,
        }
    }

    pub fn forward(&mut self, inputs: &[Array4<f32>]) -> Array4<f32> {
        println!("Route layer inputs:");
        for (i, input) in inputs.iter().enumerate() {
            println!("  Input {} shape: {:?}", i, input.shape());
        }

        // If we have multiple inputs with different spatial dimensions
        let mut tensors: Vec<Array4<f32>> = Vec::new();
        if inputs.len() > 1 {
            let h1 = inputs[0].shape()[2];
            let w1 = inputs[0].shape()[3];
            let h2 = inputs[1].shape()[2];
            let w2 = inputs[1].shape()[3];

            // If spatial dimensions don't match, create an upsampling layer
            if h1 != h2 || w1 != w2 {
                if self.upsample.is_none() {
                    // Calculate scale based on the larger dimension
                    let scale = if h1 > h2 {
                        (h1 as f32 / h2 as f32).round() as usize
                    } else {
                        (h2 as f32 / h1 as f32).round() as usize
                    };
                    println!("  Creating upsampling layer with scale {}", scale);
                    self.upsample = Some(Upsample::new(scale));
                    // Store which input needs upsampling
                    self.upsample_idx = Some(if h1 > h2 { 1 } else { 0 });
                }
            }
        }

        for (i, a) in inputs.iter().enumerate() {
            if Some(i) == self.upsample_idx {
                if let Some(ref mut up) = self.upsample {
                    tensors.push(up.forward(a));
                } else {
                    tensors.push(a.clone());
                }
            } else {
                tensors.push(a.clone());
            }
        }

        let views: Vec<_> = tensors.iter().map(|a| a.view()).collect();
        let result = concatenate(Axis(1), &views).unwrap();
        println!("Route layer output shape: {:?}", result.shape());
        result
    }
}

#[derive(Debug)]
pub struct Shortcut {
    pub from: isize,
    conv: Option<Conv2D>,  // Optional 1x1 conv for channel matching
}

impl Shortcut {
    pub fn new(from: isize) -> Self {
        Self { 
            from,
            conv: None,
        }
    }

    // Optional method to manually set the conv layer
    pub fn set_conv(&mut self, conv: Conv2D) {
        self.conv = Some(conv);
    }

    pub fn forward(&mut self, prev: &Array4<f32>, from: &Array4<f32>) -> Array4<f32> {
        println!("Shortcut layer shapes:");
        println!("  Previous shape: {:?}", prev.shape());
        println!("  From shape: {:?}", from.shape());

        // If channels don't match, create a channel-matching conv
        if prev.shape()[1] != from.shape()[1] {
            if self.conv.is_none() {
                println!("  Creating channel-matching conv: {} -> {}", from.shape()[1], prev.shape()[1]);
                self.conv = Some(Conv2D::new(
                    from.shape()[1],
                    prev.shape()[1],
                    1,  // 1x1 conv
                    1,  // stride 1
                    0,  // no padding
                    InitScheme::HeNormal
                ));
            }
        }

        // If we have a conv layer, use it to match channels
        let from = if let Some(ref mut conv) = self.conv {
            println!("  Using conv to match channels");
            conv.forward(from)
        } else {
            from.clone()
        };

        println!("  Adjusted from shape: {:?}", from.shape());
        let result = prev + from;
        println!("  Output shape: {:?}", result.shape());
        result
    }
}